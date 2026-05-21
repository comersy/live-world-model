"""

Recurrent (ConvLSTM) version. Two modes, toggled with the D key:

  LIVE  (default) - the model learns online from the webcam:
      it carries a hidden state, predicts the next frame from the current
      frame + state, compares to reality, takes one gradient step.
      Display [ REAL | PREDICTION | ERROR ].

  DREAM (press D) - the webcam is ignored. The model generates the future
      on its own: it predicts the next frame, feeds that prediction back in
      as the next input, and repeats (closed-loop autoregression), carrying
      its hidden state forward. Display [ LAST REAL FRAME | DREAM ].
      Press D again to return to LIVE.

The model predicts a residual (the change), so a static scene means delta ~ 0
and motion is what it must actually learn.

Run with:    python live.py
Quit with:   press Q   |   Toggle dream: press D

Runs on CPU. Keep the window focused so key presses are captured.
The model always starts from scratch; nothing is saved or loaded.
"""

import random
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

# === METRICS START ===  (instrumentation; delete this block when done)
import csv
import time
METRICS_PATH = "metrics.csv"
# === METRICS END ===

# --- pick which versioned model to run ---
from models.v3_convlstm.model import WorldModel
# ------------------------------------------

# ----------------------- settings -----------------------
SOURCE = "video"    # "webcam" or "video" (loops a GIF/video file forever)
VIDEO_PATH = "data/loop.gif"  # used when SOURCE == "video"
SIZE = 64            # working resolution (square, grayscale)
LR = 1e-3            # learning rate of the live step
DISPLAY_SCALE = 4    # display magnification factor
WEBCAM_INDEX = 0     # change to 1, 2... if your webcam is not index 0
TBPTT = 8            # truncated backprop window: detach state every N steps
                     # (keeps live training cheap and stable on CPU)
MOTION_WEIGHT = 20.0 # how strongly the loss focuses on moving pixels.
                     # 0 = plain MSE (v3 behavior); higher = ignore the static
                     # background more and concentrate on where motion happens.
SELF_FEED_PROB = 0.15 # during LIVE, chance to feed the model its OWN prediction
                      # instead of the real frame. Helps the dream stay alive in
                      # closed loop (mitigates, does not fully fix, drift). 0 = off.
# --------------------------------------------------------

DEVICE = torch.device("cpu")


def to_gray_small(frame: np.ndarray) -> np.ndarray:
    """BGR webcam frame -> grayscale float array [SIZE,SIZE] in [0,1]"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """float array [SIZE,SIZE] -> tensor [1,1,SIZE,SIZE]"""
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)


def pred_to_array(prediction: torch.Tensor) -> np.ndarray:
    return prediction.detach().squeeze().clamp(0, 1).cpu().numpy()


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def detach_state(state):
    return tuple(s.detach() for s in state)


class FrameSource:
    """
    Unified frame source. Yields grayscale [SIZE,SIZE] float arrays in [0,1].
      - "webcam": reads the camera live
      - "video":  loads all frames of a GIF/video once and loops them forever
    """

    def __init__(self, source, webcam_index, video_path):
        self.source = source
        if source == "webcam":
            self.cap = cv2.VideoCapture(webcam_index)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open webcam (index {webcam_index}). "
                    "Try another WEBCAM_INDEX or close apps using the camera."
                )
        elif source == "video":
            import imageio
            reader = imageio.mimread(video_path, memtest=False)
            if not reader:
                raise RuntimeError(f"No frames found in {video_path}")
            # pre-convert every frame to grayscale [SIZE,SIZE] float [0,1]
            self.frames = []
            for f in reader:
                arr = np.asarray(f)
                if arr.ndim == 3:
                    arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
                small = cv2.resize(arr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                self.frames.append(small.astype(np.float32) / 255.0)
            self.idx = 0
            print(f"Loaded {len(self.frames)} frames from {video_path} (looping)")
        else:
            raise ValueError(f"Unknown SOURCE: {source}")

    def read(self):
        """Return (ok, gray_small_float[SIZE,SIZE])."""
        if self.source == "webcam":
            ret, frame = self.cap.read()
            if not ret:
                return False, None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            return True, small.astype(np.float32) / 255.0
        else:
            arr = self.frames[self.idx]
            self.idx = (self.idx + 1) % len(self.frames)  # loop forever
            return True, arr

    def release(self):
        if self.source == "webcam":
            self.cap.release()


def motion_weighted_loss(prediction, target, prev_frame, strength):
    """
    MSE weighted by where real motion happens.

    Pixels that change between prev_frame and target get a high weight; static
    background pixels get the baseline weight of 1. This stops a mostly-static
    scene from drowning out the moving region in the average error.

    weight = 1 + strength * |target - prev_frame|
    loss   = mean( weight * (prediction - target)^2 ) / mean(weight)
    """
    motion = (target - prev_frame).abs()           # per-pixel real motion
    weight = 1.0 + strength * motion
    sq = (prediction - target) ** 2
    return (weight * sq).mean() / weight.mean()


def main():
    model = WorldModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    cap = FrameSource(SOURCE, WEBCAM_INDEX, VIDEO_PATH)

    print("live-world-model (ConvLSTM).  D = toggle dream   Q = quit")

    # prime with one real frame
    ret, cur_arr = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame.")
    cur_tensor = to_tensor(cur_arr)

    state = model.init_state(batch=1, device=DEVICE)

    last_real = cur_arr        # remembered for the dream display
    dreaming = False
    dream_steps = 0
    step = 0
    ema_loss = None

    # accumulate loss over a short window for truncated backprop
    window_loss = 0.0
    window_count = 0

    # === METRICS START ===  (instrumentation; delete this block when done)
    # Logs one row per frame to metrics.csv. Columns:
    #   t           seconds since start
    #   step        live step counter
    #   mode        "live" or "dream"
    #   loss        model MSE (live only; -1 in dream)
    #   copy_loss   MSE of the trivial "next=current" baseline (live only; -1 in dream)
    #   delta_abs   mean |predicted delta| -> ~0 means the model predicts no change
    #   real_motion mean |real_{t+1} - real_t| (live only; -1 in dream) -> how much the scene moves
    #   dream_motion mean |dream_t - dream_{t-1}| (dream only; -1 in live) -> ~0 means frozen dream
    metrics_file = open(METRICS_PATH, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(
        ["t", "step", "mode", "loss", "copy_loss",
         "delta_abs", "real_motion", "dream_motion"]
    )
    start_time = time.time()
    prev_dream_arr = None  # for dream frame-to-frame motion
    LOG_EVERY = 1          # log every frame; raise to 5 to log less often

    def mean_delta_abs(prediction_arr, input_arr):
        # predicted delta = prediction - input (the residual the model added)
        return float(np.mean(np.abs(prediction_arr - input_arr)))
    # === METRICS END ===

    while True:
        if not dreaming:
            # ---------------- LIVE MODE ----------------
            # predict next frame from current frame + carried state
            prediction, new_state = model.step(cur_tensor, state)

            # read the real next frame (the target)
            ret, target_arr = cap.read()
            if not ret:
                break
            target_tensor = to_tensor(target_arr)

            loss = motion_weighted_loss(prediction, target_tensor,
                                        cur_tensor, MOTION_WEIGHT)
            window_loss = window_loss + loss
            window_count += 1

            # truncated backprop through time: step the optimizer every TBPTT
            if window_count >= TBPTT:
                optimizer.zero_grad()
                (window_loss / window_count).backward()
                optimizer.step()
                state = detach_state(new_state)   # cut the graph, keep memory
                window_loss = 0.0
                window_count = 0
            else:
                state = new_state

            val = loss.item()
            ema_loss = val if ema_loss is None else 0.98 * ema_loss + 0.02 * val
            step += 1

            # === METRICS START ===  (instrumentation; delete this block when done)
            if step % LOG_EVERY == 0:
                pred_arr = pred_to_array(prediction)
                copy_loss = float(np.mean((cur_arr - target_arr) ** 2))   # next==current baseline
                delta_abs = mean_delta_abs(pred_arr, cur_arr)
                real_motion = float(np.mean(np.abs(target_arr - cur_arr)))
                metrics_writer.writerow(
                    [f"{time.time() - start_time:.2f}", step, "live",
                     f"{val:.6f}", f"{copy_loss:.6f}",
                     f"{delta_abs:.6f}", f"{real_motion:.6f}", -1]
                )
                metrics_file.flush()
            # === METRICS END ===

            last_real = target_arr

            # display REAL | PREDICTION | ERROR
            real_u = to_uint8(target_arr)
            pred_u = to_uint8(pred_to_array(prediction))
            err = cv2.applyColorMap(cv2.absdiff(real_u, pred_u), cv2.COLORMAP_INFERNO)
            panel = np.hstack([
                cv2.cvtColor(real_u, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(pred_u, cv2.COLOR_GRAY2BGR),
                err,
            ])
            big = cv2.resize(
                panel,
                (panel.shape[1] * DISPLAY_SCALE, panel.shape[0] * DISPLAY_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.putText(big, "REAL", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(big, "PREDICTION", (SIZE * DISPLAY_SCALE + 10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(big, "ERROR", (2 * SIZE * DISPLAY_SCALE + 10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(big, f"LIVE  step {step}  loss {ema_loss:.4f}",
                        (10, big.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("live-world-model", big)

            # next input: usually the real frame, but sometimes the model's own
            # prediction (self-feeding) so it learns to stay stable in closed loop
            if random.random() < SELF_FEED_PROB:
                fed = pred_to_array(prediction)
                cur_arr = fed
                cur_tensor = to_tensor(fed)
            else:
                cur_arr = target_arr
                cur_tensor = target_tensor

        else:
            # ---------------- DREAM MODE ----------------
            with torch.no_grad():
                prediction, state = model.step(cur_tensor, state)
            dream_arr = pred_to_array(prediction)

            # === METRICS START ===  (instrumentation; delete this block when done)
            # delta the model added this step = dream_arr - its own input
            dream_delta_abs = float(np.mean(np.abs(dream_arr - pred_to_array(cur_tensor))))
            if prev_dream_arr is None:
                dream_motion = -1.0
            else:
                dream_motion = float(np.mean(np.abs(dream_arr - prev_dream_arr)))
            metrics_writer.writerow(
                [f"{time.time() - start_time:.2f}", step, "dream",
                 -1, -1, f"{dream_delta_abs:.6f}", -1, f"{dream_motion:.6f}"]
            )
            metrics_file.flush()
            prev_dream_arr = dream_arr
            # === METRICS END ===

            # the prediction becomes the next input (closed loop)
            cur_tensor = to_tensor(dream_arr)
            dream_steps += 1

            left = cv2.cvtColor(to_uint8(last_real), cv2.COLOR_GRAY2BGR)
            right = cv2.cvtColor(to_uint8(dream_arr), cv2.COLOR_GRAY2BGR)
            panel = np.hstack([left, right])
            big = cv2.resize(
                panel,
                (panel.shape[1] * DISPLAY_SCALE, panel.shape[0] * DISPLAY_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.putText(big, "LAST REAL FRAME", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.putText(big, "DREAM", (SIZE * DISPLAY_SCALE + 10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(big, f"DREAM  step {dream_steps}   (D = back to live)",
                        (10, big.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("live-world-model", big)

            if SOURCE == "webcam":
                cap.cap.grab()  # drain webcam buffer so it isn't stale on return

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            dreaming = not dreaming
            dream_steps = 0
            window_loss = 0.0
            window_count = 0
            # === METRICS START ===  (instrumentation; delete this block when done)
            prev_dream_arr = None
            # === METRICS END ===
            if dreaming:
                print(f"--> DREAM mode (after {step} live steps)")
            else:
                print("--> LIVE mode")
                # refresh input with a real frame; keep the hidden state
                ret, cur_arr = cap.read()
                if ret:
                    cur_tensor = to_tensor(cur_arr)

    cap.release()
    cv2.destroyAllWindows()
    # === METRICS START ===  (instrumentation; delete this block when done)
    metrics_file.close()
    print(f"Metrics written to {METRICS_PATH}")
    # === METRICS END ===
    msg = f"Done after {step} live steps."
    if ema_loss is not None:
        msg += f" Final (smoothed) loss: {ema_loss:.4f}"
    print(msg)


if __name__ == "__main__":
    main()