"""
Two modes, toggled with the D key:

  LIVE  (default) - the model learns online from the webcam:
      predict next frame from the last N frames, compare to reality,
      take one gradient step. Display [ REAL | PREDICTION | ERROR ].

  DREAM (press D) - the webcam is ignored. The model generates the future
      on its own: it predicts the next frame, feeds that prediction back
      into its own memory, and repeats (closed-loop autoregression).
      Display [ LAST REAL FRAME | DREAM ]. Press D again to return to LIVE.

The point: turn your hand in a loop for a while, then press D and watch the
model keep doing the motion on its own.

To keep dreams from collapsing, during LIVE training we occasionally feed the
model its OWN prediction instead of the real frame (self-feeding). This teaches
it to stay stable when it later runs purely on its own outputs.

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

# --- pick which versioned model to run ---
from models.v2_temporal.model import WorldModel
# ------------------------------------------

# ----------------------- settings -----------------------
SIZE = 64            # working resolution (square, grayscale)
N_FRAMES = 8         # temporal memory length (must match what you trained with)
LR = 1e-3            # learning rate of the live step
DISPLAY_SCALE = 4    # display magnification factor
WEBCAM_INDEX = 0     # change to 1, 2... if your webcam is not index 0
SELF_FEED_PROB = 0.2 # during LIVE, chance to push prediction (not real frame)
                     # into memory -> trains closed-loop stability for dreaming
# --------------------------------------------------------


def to_gray_small(frame: np.ndarray) -> np.ndarray:
    """BGR webcam frame -> grayscale float array [SIZE,SIZE] in [0,1]"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def stack_to_tensor(frames: deque) -> torch.Tensor:
    """deque of N arrays [SIZE,SIZE] -> tensor [1,N,SIZE,SIZE]"""
    arr = np.stack(list(frames), axis=0)
    return torch.from_numpy(arr).unsqueeze(0)


def pred_to_array(prediction: torch.Tensor) -> np.ndarray:
    """tensor [1,1,SIZE,SIZE] in [0,1] -> float array [SIZE,SIZE]"""
    return prediction.detach().squeeze().clamp(0, 1).numpy()


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def make_panel(left_arr, right_arr, left_label, right_label, info):
    """build the side-by-side display from two float arrays [SIZE,SIZE]"""
    left = cv2.cvtColor(to_uint8(left_arr), cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(to_uint8(right_arr), cv2.COLOR_GRAY2BGR)
    panel = np.hstack([left, right])
    big = cv2.resize(
        panel,
        (panel.shape[1] * DISPLAY_SCALE, panel.shape[0] * DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.putText(big, left_label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(big, right_label, (SIZE * DISPLAY_SCALE + 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(big, info, (10, big.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return big


def main():
    device = torch.device("cpu")
    model = WorldModel(n_frames=N_FRAMES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam (index {WEBCAM_INDEX}). "
            "Try a different WEBCAM_INDEX, or close other apps using the camera."
        )

    print("live-world-model running.  D = toggle dream   Q = quit")

    # rolling window of the last N frames (the model's temporal memory)
    history = deque(maxlen=N_FRAMES)
    while len(history) < N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read frames to prime the history.")
        history.append(to_gray_small(frame))

    last_real = list(history)[-1]   # remembered for the dream display
    dreaming = False
    dream_steps = 0
    step = 0
    ema_loss = None

    while True:
        if not dreaming:
            # ---------------- LIVE MODE ----------------
            input_tensor = stack_to_tensor(history)
            prediction = model(input_tensor)

            ret, frame = cap.read()
            if not ret:
                break
            target_small = to_gray_small(frame)
            target_tensor = torch.from_numpy(target_small).unsqueeze(0).unsqueeze(0)

            # live training: one gradient step
            loss = criterion(prediction, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val = loss.item()
            ema_loss = val if ema_loss is None else 0.98 * ema_loss + 0.02 * val
            step += 1

            last_real = target_small

            # error map display
            real_u = to_uint8(target_small)
            pred_u = to_uint8(pred_to_array(prediction))
            err = cv2.applyColorMap(cv2.absdiff(real_u, pred_u),
                                    cv2.COLORMAP_INFERNO)
            real_bgr = cv2.cvtColor(real_u, cv2.COLOR_GRAY2BGR)
            pred_bgr = cv2.cvtColor(pred_u, cv2.COLOR_GRAY2BGR)
            panel = np.hstack([real_bgr, pred_bgr, err])
            big = cv2.resize(
                panel,
                (panel.shape[1] * DISPLAY_SCALE, panel.shape[0] * DISPLAY_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.putText(big, "REAL", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.putText(big, "PREDICTION", (SIZE * DISPLAY_SCALE + 10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(big, "ERROR", (2 * SIZE * DISPLAY_SCALE + 10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(big, f"LIVE  step {step}  loss {ema_loss:.4f}",
                        (10, big.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.imshow("live-world-model", big)

            # slide the window: usually the real frame, sometimes the
            # prediction (self-feeding) to train closed-loop stability
            if random.random() < SELF_FEED_PROB:
                history.append(pred_to_array(prediction))
            else:
                history.append(target_small)

        else:
            # ---------------- DREAM MODE ----------------
            # the webcam is ignored; the model runs on its own outputs
            input_tensor = stack_to_tensor(history)
            with torch.no_grad():
                prediction = model(input_tensor)
            dream_arr = pred_to_array(prediction)

            # feed the prediction back into memory (closed loop)
            history.append(dream_arr)
            dream_steps += 1

            big = make_panel(
                last_real, dream_arr,
                "LAST REAL FRAME", "DREAM",
                f"DREAM  step {dream_steps}   (D = back to live)",
            )
            cv2.imshow("live-world-model", big)

            # drain webcam buffer so it isn't stale when we return to live
            cap.grab()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            dreaming = not dreaming
            dream_steps = 0
            if dreaming:
                print(f"--> DREAM mode (after {step} live steps)")
            else:
                print("--> LIVE mode")
                # re-prime memory with fresh real frames after dreaming
                history.clear()
                while len(history) < N_FRAMES:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    history.append(to_gray_small(frame))

    cap.release()
    cv2.destroyAllWindows()
    msg = f"Done after {step} live steps."
    if ema_loss is not None:
        msg += f" Final (smoothed) loss: {ema_loss:.4f}"
    print(msg)


if __name__ == "__main__":
    main()