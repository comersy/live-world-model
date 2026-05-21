"""
live-world-model / live.py

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

from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

# --- pick which versioned model to run ---
from models.v3_convlstm.model import WorldModel
# ------------------------------------------

# ----------------------- settings -----------------------
SIZE = 64            # working resolution (square, grayscale)
LR = 1e-3            # learning rate of the live step
DISPLAY_SCALE = 4    # display magnification factor
WEBCAM_INDEX = 0     # change to 1, 2... if your webcam is not index 0
TBPTT = 8            # truncated backprop window: detach state every N steps
                     # (keeps live training cheap and stable on CPU)
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


def main():
    model = WorldModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam (index {WEBCAM_INDEX}). "
            "Try a different WEBCAM_INDEX, or close other apps using the camera."
        )

    print("live-world-model (ConvLSTM).  D = toggle dream   Q = quit")

    # prime with one real frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame.")
    cur_arr = to_gray_small(frame)
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

    while True:
        if not dreaming:
            # ---------------- LIVE MODE ----------------
            # predict next frame from current frame + carried state
            prediction, new_state = model.step(cur_tensor, state)

            # read the real next frame (the target)
            ret, frame = cap.read()
            if not ret:
                break
            target_arr = to_gray_small(frame)
            target_tensor = to_tensor(target_arr)

            loss = criterion(prediction, target_tensor)
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

            # the real frame becomes the next input
            cur_arr = target_arr
            cur_tensor = target_tensor

        else:
            # ---------------- DREAM MODE ----------------
            with torch.no_grad():
                prediction, state = model.step(cur_tensor, state)
            dream_arr = pred_to_array(prediction)

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

            cap.grab()  # drain webcam buffer so it isn't stale on return

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            dreaming = not dreaming
            dream_steps = 0
            window_loss = 0.0
            window_count = 0
            if dreaming:
                print(f"--> DREAM mode (after {step} live steps)")
            else:
                print("--> LIVE mode")
                # refresh input with a real frame; keep the hidden state
                ret, frame = cap.read()
                if ret:
                    cur_arr = to_gray_small(frame)
                    cur_tensor = to_tensor(cur_arr)

    cap.release()
    cv2.destroyAllWindows()
    msg = f"Done after {step} live steps."
    if ema_loss is not None:
        msg += f" Final (smoothed) loss: {ema_loss:.4f}"
    print(msg)


if __name__ == "__main__":
    main()