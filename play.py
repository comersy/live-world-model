"""

Watch the OFFLINE-TRAINED v5 model on the recorded loop. No training here — the
model is frozen (already trained by train_offline.py), this is just for judging.

Two modes, toggled with D:

  PREDICT (default) - the real loop plays; for each frame the model predicts the
      next one. Display [ REAL | PREDICTION | ERROR ].

  DREAM (press D) - the loop is ignored; the model runs on its own predictions
      in closed loop. The hidden state carries over from where you pressed D.
      Display [ REAL (frozen seed) | PREDICTION (dream) | ERROR vs that seed ].
      Press D again to go back to PREDICT (re-seeds the state on the loop).

Keys:  D = toggle dream    Q = quit

"""

import cv2
import numpy as np
import imageio
import torch

from models.v4_offline.model import WorldModel

# ----------------------- settings -----------------------
GIF_PATH = "data/loop.gif"
WEIGHTS = "models/v4_offline/weights.pt"
SIZE = 64
DISPLAY_SCALE = 4
FPS = 15
# --------------------------------------------------------

DEVICE = torch.device("cpu")


def load_loop(path):
    frames = imageio.mimread(path, memtest=False)
    gs = []
    for f in frames:
        a = np.asarray(f)
        if a.ndim == 3:
            a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2GRAY)
        small = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        gs.append(small.astype(np.float32) / 255.0)
    return np.array(gs)


def to_tensor(arr):
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)


def to_uint8(arr):
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def main():
    model = WorldModel().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()
    loop = load_loop(GIF_PATH)
    T = len(loop)
    print(f"Loaded model + {T} loop frames.  D = toggle dream   Q = quit")

    state = model.init_state(1, DEVICE)
    idx = 0                      # position in the loop
    cur = to_tensor(loop[0])     # current input frame
    dreaming = False
    dream_seed = loop[0]         # frozen "real" frame shown during dream
    dream_step = 0

    while True:
        with torch.no_grad():
            pred, state = model.step(cur, state)
        pred_arr = pred.squeeze().clamp(0, 1).cpu().numpy()

        if not dreaming:
            # PREDICT MODE: compare prediction to the real next loop frame
            nxt = loop[(idx + 1) % T]
            real_shown = nxt
            err_ref = nxt
            # advance along the real loop; next input is the real next frame
            idx = (idx + 1) % T
            cur = to_tensor(nxt)
            info = f"PREDICT  frame {idx}/{T}   (D = dream)"
        else:
            # DREAM MODE: feed prediction back; show the frozen seed on the left
            real_shown = dream_seed
            err_ref = dream_seed
            cur = pred  # closed loop
            dream_step += 1
            info = f"DREAM  step {dream_step}   (D = back)"

        # build [ REAL | PREDICTION | ERROR ]
        real_u = to_uint8(real_shown)
        pred_u = to_uint8(pred_arr)
        err = cv2.applyColorMap(cv2.absdiff(to_uint8(err_ref), pred_u),
                                cv2.COLORMAP_INFERNO)
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
        labels = ["REAL", "PREDICTION", "ERROR"]
        if dreaming:
            labels = ["REAL (seed)", "DREAM", "ERROR vs seed"]
        for j, lab in enumerate(labels):
            cv2.putText(big, lab, (j * SIZE * DISPLAY_SCALE + 10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(big, info, (10, big.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("live-world-model", big)

        key = cv2.waitKey(int(1000 / FPS)) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            dreaming = not dreaming
            if dreaming:
                dream_seed = real_shown   # freeze current real frame as the seed
                dream_step = 0
                print("--> DREAM")
            else:
                # re-seed the hidden state on the real loop for clean prediction
                print("--> PREDICT")
                state = model.init_state(1, DEVICE)
                with torch.no_grad():
                    for t in range(min(40, T - 1)):
                        _, state = model.step(to_tensor(loop[t]), state)
                idx = min(40, T - 1)
                cur = to_tensor(loop[idx])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()