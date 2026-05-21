"""
offline / play.py

Load the trained model and watch it on the recorded loop. No training (frozen).

  PREDICT mode: the loop plays; model predicts the next frame.
                display [ REAL | PREDICTION | ERROR ]
  DREAM mode (press D): model runs on its own predictions (closed loop).
                display [ REAL (seed) | DREAM | ERROR vs seed ]

Keys: D = toggle dream, Q = quit

Run from the project root:   python -m offline.play
"""

import cv2
import numpy as np
import torch

from core.models.v5_hires.model import WorldModel, SIZE
from core.data_io import load_loop, to_tensor, to_uint8

WEIGHTS = "weights/v5_hires.pt"
GIF_PATH = "data/loop.gif"
DISPLAY_SCALE = 3
FPS = 15
DEVICE = torch.device("cpu")


def main():
    model = WorldModel().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()
    loop = load_loop(GIF_PATH, SIZE).squeeze(1).numpy()  # [T,SIZE,SIZE]
    T = len(loop)
    print(f"Loaded model + {T} frames. D = dream, Q = quit")

    state = model.init_state(1, DEVICE)
    idx = 0
    cur = to_tensor(loop[0], DEVICE)
    dreaming = False
    seed = loop[0]
    dstep = 0

    while True:
        with torch.no_grad():
            pred, state = model.step(cur, state)
        pred_arr = pred.squeeze().clamp(0, 1).cpu().numpy()

        if not dreaming:
            nxt = loop[(idx + 1) % T]
            shown, ref = nxt, nxt
            idx = (idx + 1) % T
            cur = to_tensor(nxt, DEVICE)
            info = f"PREDICT {idx}/{T}  (D=dream)"
            labels = ["REAL", "PREDICTION", "ERROR"]
        else:
            shown, ref = seed, seed
            cur = pred
            dstep += 1
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
        cv2.imshow("offline play", big)

        key = cv2.waitKey(int(1000 / FPS)) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            dreaming = not dreaming
            if dreaming:
                seed = shown
                dstep = 0
            else:
                state = model.init_state(1, DEVICE)
                with torch.no_grad():
                    for t in range(min(40, T - 1)):
                        _, state = model.step(to_tensor(loop[t], DEVICE), state)
                idx = min(40, T - 1)
                cur = to_tensor(loop[idx], DEVICE)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()