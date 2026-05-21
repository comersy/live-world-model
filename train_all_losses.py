"""

Train the SAME model on the recorded loop with several different losses, in one
run. Each loss produces its own weights file in models/<ver>/experiments/, so
they can be compared fairly afterwards with compare.py.

All losses are trained under identical conditions (same loop, same epochs, same
seed) — only the loss function differs. This makes the comparison meaningful.

Losses implemented:
  mse      - plain pixel MSE (the baseline)
  l1       - pixel L1 (mean abs error); blurs less than MSE
  grad     - gradient/edge loss: matches image gradients (fights blur directly)
  l1_grad  - L1 + gradient loss combined

Run with:   python train_all_losses.py
"""

import time

import cv2
import numpy as np
import imageio
import torch

# NOTE: in your project the folder is "v4_offline"; adapt this import to match.
from models.v4_offline.model import WorldModel

# ----------------------- settings -----------------------
GIF_PATH = "data/loop.gif"
OUT_DIR = "models/v5_offline/experiments"   # adapt to v4_offline in your repo
SIZE = 64
LR = 1e-3
EPOCHS = 150                 # comparison phase; raise to ~400 for the winner
TF_CHUNK = 20
ROLLOUT_EVERY = 3
ROLLOUT_LEN = 12
LOSSES = ["mse", "l1", "grad", "l1_grad"]
SEED = 0
LOG_EVERY = 50
# --------------------------------------------------------

DEVICE = torch.device("cpu")


# ---------------------- loss functions ----------------------
def image_gradients(x):
    """return (dx, dy) finite differences of a [B,1,H,W] image"""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def grad_loss(pred, target):
    px, py = image_gradients(pred)
    tx, ty = image_gradients(target)
    return (px - tx).abs().mean() + (py - ty).abs().mean()


def compute_loss(name, pred, target):
    if name == "mse":
        return ((pred - target) ** 2).mean()
    if name == "l1":
        return (pred - target).abs().mean()
    if name == "grad":
        # keep a little pixel term so it doesn't drift in absolute brightness
        return 0.5 * (pred - target).abs().mean() + grad_loss(pred, target)
    if name == "l1_grad":
        return (pred - target).abs().mean() + grad_loss(pred, target)
    raise ValueError(name)
# ------------------------------------------------------------


def load_loop(path):
    frames = imageio.mimread(path, memtest=False)
    gs = []
    for f in frames:
        a = np.asarray(f)
        if a.ndim == 3:
            a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2GRAY)
        small = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        gs.append(small.astype(np.float32) / 255.0)
    return torch.from_numpy(np.array(gs)).unsqueeze(1).to(DEVICE)


def train_one(loss_name, seq):
    T = seq.shape[0]
    torch.manual_seed(SEED)  # same init for every loss -> fair comparison
    model = WorldModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(EPOCHS):
        do_rollout = (epoch % ROLLOUT_EVERY == 0)
        state = model.init_state(1, DEVICE)
        opt.zero_grad()
        running = 0.0
        chunk = 0
        t = 0
        while t < T - 1:
            if do_rollout and t + ROLLOUT_LEN < T:
                inp = seq[t:t + 1]
                rl = 0.0
                for k in range(ROLLOUT_LEN):
                    pred, state = model.step(inp, state)
                    rl = rl + compute_loss(loss_name, pred, seq[t + k + 1:t + k + 2])
                    inp = pred
                (rl / ROLLOUT_LEN).backward()
                opt.step()
                opt.zero_grad()
                state = tuple(s.detach() for s in state)
                t += ROLLOUT_LEN
            else:
                pred, state = model.step(seq[t:t + 1], state)
                running = running + compute_loss(loss_name, pred, seq[t + 1:t + 2])
                chunk += 1
                if chunk >= TF_CHUNK or t == T - 2:
                    running.backward()
                    opt.step()
                    opt.zero_grad()
                    state = tuple(s.detach() for s in state)
                    running = 0.0
                    chunk = 0
                t += 1
        if epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1:
            print(f"  [{loss_name}] epoch {epoch:3d}  ({time.time() - t0:.0f}s)")

    out = f"{OUT_DIR}/weights_{loss_name}.pt"
    torch.save(model.state_dict(), out)
    print(f"  [{loss_name}] saved -> {out}")


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    seq = load_loop(GIF_PATH)
    print(f"Loaded {seq.shape[0]} frames. Training {len(LOSSES)} losses x {EPOCHS} epochs.")
    total = time.time()
    for name in LOSSES:
        print(f"\n=== training loss: {name} ===")
        train_one(name, seq)
    print(f"\nAll done in {time.time() - total:.0f}s. Compare with: python compare.py")


if __name__ == "__main__":
    main()