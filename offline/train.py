"""

Train the world model OFFLINE on a recorded loop (data/loop.gif), going over it
many times. Not real-time. Saves weights to weights/v5_hires.pt.

Uses:
  - motion-weighted loss (focus on the moving region; MOTION_WEIGHT)
  - rollout training every few epochs (closed-loop, fixes dream collapse)

Run from the project root:   python -m offline.train
"""

import time
import torch

from core.models.v5_hires.model import WorldModel, SIZE
from core.data_io import load_loop
from core.losses import motion_weighted_loss

# ----------------------- settings -----------------------
GIF_PATH = "data/loop.gif"
WEIGHTS_OUT = "weights/v5_hires.pt"
LR = 1e-3
EPOCHS = 300
MOTION_WEIGHT = 50.0     # how much moving pixels matter (0 = plain MSE)
TF_CHUNK = 20
ROLLOUT_EVERY = 3
ROLLOUT_LEN = 24
LOG_EVERY = 25
# --------------------------------------------------------

DEVICE = torch.device("cpu")


def main():
    seq = load_loop(GIF_PATH, SIZE).to(DEVICE)
    T = seq.shape[0]
    print(f"Loaded {T} frames at {SIZE}x{SIZE} from {GIF_PATH}")

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
                inp = seq[t:t + 1]
                roll = 0.0
                for k in range(ROLLOUT_LEN):
                    pred, state = model.step(inp, state)
                    target = seq[t + k + 1:t + k + 2]
                    roll = roll + motion_weighted_loss(pred, target, inp, MOTION_WEIGHT)
                    inp = pred
                (roll / ROLLOUT_LEN).backward()
                opt.step()
                opt.zero_grad()
                state = tuple(s.detach() for s in state)
                losses.append((roll / ROLLOUT_LEN).item())
                t += ROLLOUT_LEN
            else:
                prev = seq[t:t + 1]
                target = seq[t + 1:t + 2]
                pred, state = model.step(prev, state)
                l = motion_weighted_loss(pred, target, prev, MOTION_WEIGHT)
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
            tag = "rollout" if do_rollout else "teacher"
            print(f"epoch {epoch:3d} [{tag}]  loss={sum(losses)/len(losses):.6f}  "
                  f"({time.time() - t0:.0f}s)")

    import os
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_OUT)
    print(f"\nSaved weights to {WEIGHTS_OUT}")


if __name__ == "__main__":
    main()