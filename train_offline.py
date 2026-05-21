"""

Train the model OFFLINE on a recorded loop (a GIF), going over the whole
sequence many times. This is far more effective than live training: in live
mode the model only sees each part of the loop a few times per minute, which
is not enough. Offline we can do hundreds of passes in a couple of minutes.

It also uses a bit of ROLLOUT during training: every few epochs the model is
forced to predict several steps from its OWN predictions (closed loop), and is
penalized on the whole rollout. This is the proper fix for the dream collapsing
(exposure bias): the model learns to stay coherent when running on its own
outputs, which is exactly what dreaming does.

When done, weights are saved to the v5 folder. Then run live.py (pointed at v5)
to dream from the trained model.
"""

import time

import cv2
import numpy as np
import imageio
import torch

from models.v4_offline.model import WorldModel

# ----------------------- settings -----------------------
GIF_PATH = "data/loop.gif"   # the recorded loop to train on
WEIGHTS_OUT = "models/v4_offline/weights.pt"
SIZE = 64
LR = 1e-3
MOTION_WEIGHT = 100.0   # how much moving pixels matter vs the static background
EPOCHS = 400                 # passes over the whole loop
TF_CHUNK = 20                # teacher-forcing chunk for truncated backprop
ROLLOUT_EVERY = 3            # every N epochs, do a closed-loop rollout pass
ROLLOUT_LEN = 24             # how many steps to roll out on own predictions
LOG_EVERY = 25
# --------------------------------------------------------

DEVICE = torch.device("cpu")



def load_loop(path):
    frames = imageio.mimread(path, memtest=False)
    if not frames:
        raise RuntimeError(f"No frames found in {path}")
    gs = []
    for f in frames:
        a = np.asarray(f)
        if a.ndim == 3:
            a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2GRAY)
        small = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        gs.append(small.astype(np.float32) / 255.0)
    seq = torch.from_numpy(np.array(gs)).unsqueeze(1).to(DEVICE)  # [T,1,64,64]
    return seq

def motion_loss(pred, target, prev):
    # weight each pixel by how much it really moves between prev and target
    motion = (target - prev).abs()
    weight = 1.0 + MOTION_WEIGHT * motion
    sq = (pred - target) ** 2
    return (weight * sq).mean() / weight.mean()


def main():
    seq = load_loop(GIF_PATH)
    T = seq.shape[0]
    print(f"Loaded {T} frames from {GIF_PATH}")

    # copy baseline: MSE of predicting next == current, over the loop
    copy_mse = float(((seq[1:] - seq[:-1]) ** 2).mean())
    print(f"Copy baseline MSE: {copy_mse:.6f}  (model should beat this)")

    model = WorldModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(EPOCHS):
        do_rollout = (epoch % ROLLOUT_EVERY == 0)

        state = model.init_state(1, DEVICE)
        opt.zero_grad()
        running = 0.0
        chunk = 0
        losses = []

        t = 0
        while t < T - 1:
            if do_rollout and t + ROLLOUT_LEN < T:
                # closed-loop rollout: predict ROLLOUT_LEN steps from own output
                inp = seq[t:t + 1]
                roll_loss = 0.0
                for k in range(ROLLOUT_LEN):
                    pred, state = model.step(inp, state)
                    target = seq[t + k + 1:t + k + 2]
                    roll_loss = roll_loss + motion_loss(pred, target, inp)
                    inp = pred  # feed own prediction back (like dreaming)
                roll_loss = roll_loss / ROLLOUT_LEN
                roll_loss.backward()
                opt.step()
                opt.zero_grad()
                state = tuple(s.detach() for s in state)
                losses.append(roll_loss.item())
                t += ROLLOUT_LEN
            else:
                # teacher forcing: predict one step from the real frame
                pred, state = model.step(seq[t:t + 1], state)
                l = motion_loss(pred, seq[t + 1:t + 2], seq[t:t + 1])
                running = running + l
                chunk += 1
                losses.append(l.item())
                if chunk >= TF_CHUNK or t == T - 2:
                    running.backward()
                    opt.step()
                    opt.zero_grad()
                    state = tuple(s.detach() for s in state)
                    running = 0.0
                    chunk = 0
                t += 1

        if epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1:
            ml = float(np.mean(losses))
            tag = "rollout" if do_rollout else "teacher"
            print(f"epoch {epoch:3d} [{tag}]  loss={ml:.6f}  "
                  f"ratio_vs_copy={ml / copy_mse:.2f}  ({time.time() - t0:.1f}s)")

    torch.save(model.state_dict(), WEIGHTS_OUT)
    print(f"\nSaved trained weights to {WEIGHTS_OUT}")
    print("Now run live.py (set to the v4 model + load these weights) to dream.")


if __name__ == "__main__":
    main()