"""
live-world-model / live.py

The living loop. On each iteration:
  1. capture the current webcam frame
  2. the model predicts the NEXT frame from the last N frames
  3. wait for the real next frame
  4. compute the error and take ONE gradient step (this is the live training)
  5. display  [ REAL | PREDICTION | ERROR ]

The model always starts from scratch: it learns only from the current live
session. Nothing is saved or loaded.

Run with:    python live.py
Quit with:   press Q (in the display window)

Runs on CPU. Keep the window focused so key presses are captured.
"""

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
N_FRAMES = 4         # how many past frames the model sees (temporal memory)
LR = 1e-3            # learning rate of the live step
DISPLAY_SCALE = 4    # display magnification factor
WEBCAM_INDEX = 0     # change to 1, 2... if your webcam is not index 0
# --------------------------------------------------------


def to_gray_small(frame: np.ndarray) -> np.ndarray:
    """BGR webcam frame -> grayscale float array [SIZE,SIZE] in [0,1]"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def stack_to_tensor(frames: deque) -> torch.Tensor:
    """deque of N arrays [SIZE,SIZE] -> tensor [1,N,SIZE,SIZE]"""
    arr = np.stack(list(frames), axis=0)          # [N,SIZE,SIZE]
    return torch.from_numpy(arr).unsqueeze(0)     # [1,N,SIZE,SIZE]


def to_image(tensor: torch.Tensor) -> np.ndarray:
    """tensor [1,1,SIZE,SIZE] in [0,1] -> grayscale uint8 image [SIZE,SIZE]"""
    arr = tensor.detach().squeeze().clamp(0, 1).numpy()
    return (arr * 255).astype(np.uint8)


def main():
    device = torch.device("cpu")
    model = WorldModel(n_frames=N_FRAMES).to(device)  # always from scratch
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam (index {WEBCAM_INDEX}). "
            "Try a different WEBCAM_INDEX, or make sure no other "
            "application is using the camera."
        )

    print("live-world-model running. Focus the window + press Q to quit.")

    # rolling window of the last N frames (the model's temporal memory)
    history = deque(maxlen=N_FRAMES)

    # prime the history with the first N frames before predicting
    while len(history) < N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read frames to prime the history.")
        history.append(to_gray_small(frame))

    step = 0
    ema_loss = None  # smoothed loss for a stable on-screen readout

    while True:
        # predict the next frame from the last N frames
        input_tensor = stack_to_tensor(history)
        prediction = model(input_tensor)

        # capture the real next frame (the target)
        ret, frame = cap.read()
        if not ret:
            break
        target_small = to_gray_small(frame)
        target_tensor = torch.from_numpy(target_small).unsqueeze(0).unsqueeze(0)

        # --- live training: one gradient step ---
        loss = criterion(prediction, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val = loss.item()
        ema_loss = val if ema_loss is None else 0.98 * ema_loss + 0.02 * val
        step += 1

        # --- side-by-side display: real | prediction | error ---
        real_img = (target_small * 255).astype(np.uint8)
        pred_img = to_image(prediction)
        err_img = cv2.absdiff(real_img, pred_img)
        err_img = cv2.applyColorMap(err_img, cv2.COLORMAP_INFERNO)

        real_bgr = cv2.cvtColor(real_img, cv2.COLOR_GRAY2BGR)
        pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)

        panel = np.hstack([real_bgr, pred_bgr, err_img])
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
        cv2.putText(big, f"step {step}  loss {ema_loss:.4f}",
                    (10, big.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.imshow("live-world-model", big)

        # slide the window forward: the real frame joins the history
        history.append(target_small)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done after {step} steps. Final (smoothed) loss: {ema_loss:.4f}")


if __name__ == "__main__":
    main()