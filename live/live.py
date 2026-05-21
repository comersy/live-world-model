"""
live / live.py

The goal: a live webcam world model. It learns online from the webcam (one
gradient step per frame, motion-weighted loss), and on D it dreams — running on
its own predictions in closed loop.

  PREDICT/learn: display [ REAL | PREDICTION | ERROR ]
  D : toggle dream  [ REAL (seed) | DREAM | ERROR vs seed ]
  Q : quit

Optionally starts from offline-trained weights (PRETRAINED) so it begins
already competent, then keeps adapting live.

Run from the project root:   python -m live.live
"""

import os
import cv2
import numpy as np
import torch

from core.models.v5_hires.model import WorldModel, SIZE
from core.data_io import frame_to_gray, to_tensor, to_uint8
from core.losses import motion_weighted_loss

# ----------------------- settings -----------------------
WEBCAM_INDEX = 0
DISPLAY_SCALE = 3
LR = 1e-3
MOTION_WEIGHT = 50.0
PRETRAINED = "weights/v5_hires.pt"  # set to "" to start from scratch
# --------------------------------------------------------

DEVICE = torch.device("cpu")


def main():
    model = WorldModel().to(DEVICE)
    if PRETRAINED and os.path.exists(PRETRAINED):
        model.load_state_dict(torch.load(PRETRAINED, map_location=DEVICE))
        print(f"Loaded pretrained weights from {PRETRAINED}")
    else:
        print("Starting from scratch (no pretrained weights).")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {WEBCAM_INDEX}).")
    print("D = toggle dream, Q = quit")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")
    cur_arr = frame_to_gray(frame, SIZE)
    cur = to_tensor(cur_arr, DEVICE)
    state = model.init_state(1, DEVICE)

    dreaming = False
    seed = cur_arr
    dstep = 0

    while True:
        if not dreaming:
            # predict next frame from current + learn against the real next
            pred, new_state = model.step(cur, state)
            ret, frame = cap.read()
            if not ret:
                break
            target_arr = frame_to_gray(frame, SIZE)
            target = to_tensor(target_arr, DEVICE)

            loss = motion_weighted_loss(pred, target, cur, MOTION_WEIGHT)
            opt.zero_grad()
            loss.backward()
            opt.step()
            state = tuple(s.detach() for s in new_state)

            pred_arr = pred.squeeze().clamp(0, 1).cpu().detach().numpy()
            shown, ref = target_arr, target_arr
            seed = target_arr
            cur_arr = target_arr
            cur = target
            info = f"LIVE  loss {loss.item():.4f}  (D=dream)"
            labels = ["REAL", "PREDICTION", "ERROR"]
        else:
            with torch.no_grad():
                pred, state = model.step(cur, state)
            pred_arr = pred.squeeze().clamp(0, 1).cpu().numpy()
            shown, ref = seed, seed
            cur = pred
            dstep += 1
            cap.grab()
            info = f"DREAM {dstep}  (D=back)"
            labels = ["REAL (seed)", "DREAM", "ERROR vs seed"]

        err = cv2.applyColorMap(cv2.absdiff(to_uint8(ref), to_uint8(pred_arr)),
                                cv2.COLORMAP_INFERNO)
        panel = np.hstack([
            cv2.cvtColor(to_uint8(shown), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(to_uint8(pred_arr), cv2.COLOR_GRAY2BGR),
            err,
        ])
        big = cv2.resize(panel, (panel.shape[1] * DISPLAY_SCALE, panel.shape[0] * DISPLAY_SCALE),
                         interpolation=cv2.INTER_NEAREST)
        for j, lab in enumerate(labels):
            cv2.putText(big, lab, (j * SIZE * DISPLAY_SCALE + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(big, info, (8, big.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("live-world-model", big)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            dreaming = not dreaming
            if dreaming:
                seed = shown
                dstep = 0
                print("--> DREAM")
            else:
                print("--> LIVE")
                ret, frame = cap.read()
                if ret:
                    cur_arr = frame_to_gray(frame, SIZE)
                    cur = to_tensor(cur_arr, DEVICE)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()