"""

Load every weights_<loss>.pt produced by train_all_losses.py, dream with each
on the recorded loop, and report two metrics so the losses can be compared:

  motion_survival : dream_motion(last) / dream_motion(first).
                    ~1.0 = motion holds; ~0 = the dream freezes.
  sharpness_keep  : edge energy(last) / edge energy(first) of the dream.
                    ~1.0 = stays as sharp as it started; <1 = blurs out.

Then it shows each dream visually, one after another (press any key to advance,
Q to skip the rest).

Run with:   python compare.py
"""

import glob
import os

import cv2
import numpy as np
import imageio
import torch

# NOTE: adapt to v4_offline in your repo.
from models.v5_offline.model import WorldModel

GIF_PATH = "data/loop.gif"
EXP_DIR = "models/v5_offline/experiments"
SIZE = 64
DISPLAY_SCALE = 4
SEED_FRAMES = 40
DREAM_STEPS = 120
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


def edge_energy(img):
    """mean absolute gradient — a simple sharpness proxy"""
    dx = np.abs(img[:, 1:] - img[:, :-1]).mean()
    dy = np.abs(img[1:, :] - img[:-1, :]).mean()
    return float(dx + dy)


def dream_run(model, loop, collect_frames=False):
    state = model.init_state(1, DEVICE)
    with torch.no_grad():
        for t in range(min(SEED_FRAMES, len(loop) - 1)):
            _, state = model.step(to_tensor(loop[t]), state)
        cur = to_tensor(loop[min(SEED_FRAMES, len(loop) - 1)])
        motions, edges, frames = [], [], []
        prev = None
        for _ in range(DREAM_STEPS):
            pred, state = model.step(cur, state)
            d = pred.squeeze().clamp(0, 1).cpu().numpy()
            if prev is not None:
                motions.append(float(np.mean(np.abs(d - prev))))
            edges.append(edge_energy(d))
            if collect_frames:
                frames.append(d)
            prev = d
            cur = pred
    return motions, edges, frames


def main():
    loop = load_loop(GIF_PATH)
    # reference sharpness: average edge energy of the real loop frames
    real_sharp = float(np.mean([edge_energy(f) for f in loop]))

    paths = sorted(glob.glob(f"{EXP_DIR}/weights_*.pt"))
    if not paths:
        print(f"No weights found in {EXP_DIR}. Run train_all_losses.py first.")
        return

    print(f"real-loop edge energy (reference): {real_sharp:.4f}\n")
    print(f"{'loss':<12}{'motion_survival':>16}{'sharpness_vs_real':>18}")
    print("-" * 46)
    results = {}
    for p in paths:
        name = os.path.basename(p)[len("weights_"):-len(".pt")]
        model = WorldModel().to(DEVICE)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        model.eval()
        motions, edges, _ = dream_run(model, loop)
        surv = motions[-1] / motions[0] if motions and motions[0] > 0 else 0.0
        # sharpness of the LAST dream frame relative to the real loop:
        #   ~1.0 = as sharp as reality, <1 = blurrier, >1 = noisier/over-sharp
        sharp = edges[-1] / real_sharp if real_sharp > 0 else 0.0
        results[name] = p
        print(f"{name:<12}{surv:>16.2f}{sharp:>18.2f}")
    print("-" * 46)
    print("motion_survival ~1 = motion holds (higher better).")
    print("sharpness_vs_real ~1 = as sharp as the real loop; <1 blurry, >1 noisy.")

    # visual pass
    print("\nShowing each dream. Any key = next, Q = stop.")
    for name, p in results.items():
        model = WorldModel().to(DEVICE)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        model.eval()
        _, _, frames = dream_run(model, loop, collect_frames=True)
        stop = False
        for i, d in enumerate(frames):
            img = cv2.cvtColor((np.clip(d, 0, 1) * 255).astype(np.uint8),
                               cv2.COLOR_GRAY2BGR)
            big = cv2.resize(img, (SIZE * DISPLAY_SCALE, SIZE * DISPLAY_SCALE),
                             interpolation=cv2.INTER_NEAREST)
            cv2.putText(big, f"{name}  dream {i}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("compare", big)
            k = cv2.waitKey(60) & 0xFF
            if k == ord("q"):
                stop = True
                break
        if not stop:
            cv2.waitKey(0)  # pause on last frame until a key
        if stop:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()