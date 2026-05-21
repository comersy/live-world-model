"""

Train the world model (generator) on the recorded loop with a COMBINED loss:

    generator_loss = MSE(pred, real_next)  +  ADV_WEIGHT * adversarial_term

The MSE keeps the model anchored on the right dynamics (and stops the GAN from
flying off the rails). The adversarial term, from a discriminator that judges
(prev, next) frame PAIRS, pushes predictions to be sharp AND in motion — since
the discriminator rejects both blur and frozen frames.

This is the pix2pix-style recipe (reconstruction loss + adversarial loss),
which is far more stable than a pure GAN.

Output: generator weights saved to the v5_gan folder. Discriminator is discarded.

GANs are finicky. If the dream collapses or looks noisy, the knobs to try are
ADV_WEIGHT (lower = safer, more MSE-like), and the learning rates.

"""

import time

import cv2
import numpy as np
import imageio
import torch
import torch.nn as nn

from models.v5_gan.model import WorldModel, Discriminator

# ----------------------- settings -----------------------
GIF_PATH = "data/loop.gif"
WEIGHTS_OUT = "models/v5_gan/weights_gan.pt"
SIZE = 64
EPOCHS = 200
G_LR = 5e-4          # generator learning rate
D_LR = 1e-4          # discriminator LR (lower so D doesn't overpower G)
ADV_WEIGHT = 0.3     # weight of the adversarial term (raised to wake up the GAN)
D_EVERY = 2          # train the discriminator only every N steps (slows it down)
PRETRAIN = "models/v4_offline/weights.pt"  # start G from v4/v5 MSE weights; adapt path
TF_CHUNK = 20
ROLLOUT_EVERY = 3
ROLLOUT_LEN = 12
SEED = 0
LOG_EVERY = 25
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
    return torch.from_numpy(np.array(gs)).unsqueeze(1).to(DEVICE)


def main():
    torch.manual_seed(SEED)
    seq = load_loop(GIF_PATH)
    T = seq.shape[0]
    print(f"Loaded {T} frames. GAN training for {EPOCHS} epochs.")

    G = WorldModel().to(DEVICE)
    # pretraining: start the generator from the MSE-trained weights so the GAN
    # only has to refine, not learn from scratch (keeps the match balanced)
    import os
    if os.path.exists(PRETRAIN):
        G.load_state_dict(torch.load(PRETRAIN, map_location=DEVICE))
        print(f"Loaded pretrained generator from {PRETRAIN}")
    else:
        print(f"WARNING: no pretrained weights at {PRETRAIN}, starting G from scratch")
    D = Discriminator().to(DEVICE)
    g_opt = torch.optim.Adam(G.parameters(), lr=G_LR, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=D_LR, betas=(0.5, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    t0 = time.time()
    for epoch in range(EPOCHS):
        state = G.init_state(1, DEVICE)
        d_losses, g_adv_losses, mse_losses = [], [], []

        t = 0
        while t < T - 1:
            prev = seq[t:t + 1]
            real_next = seq[t + 1:t + 2]

            # ---- generator forward (one step) ----
            pred, new_state = G.step(prev, state)

            # ===== train discriminator (real vs fake transition) =====
            # only every D_EVERY steps, so D doesn't outpace G
            if t % D_EVERY == 0:
                d_opt.zero_grad()
                real_logit = D(prev, real_next)
                fake_logit = D(prev, pred.detach())   # detach: don't touch G here
                d_loss = (bce(real_logit, torch.ones_like(real_logit)) +
                          bce(fake_logit, torch.zeros_like(fake_logit))) * 0.5
                d_loss.backward()
                d_opt.step()
                d_losses.append(d_loss.item())

            # ===== train generator: MSE + adversarial, each step =====
            g_logit = D(prev, pred)                       # G wants D to say "real"
            adv = bce(g_logit, torch.ones_like(g_logit))  # fool D
            recon = mse(pred, real_next)
            g_step_loss = recon + ADV_WEIGHT * adv
            g_opt.zero_grad()
            g_step_loss.backward()
            g_opt.step()
            mse_losses.append(recon.item())
            g_adv_losses.append(adv.item())

            # carry the hidden state forward, detached (truncated BPTT, length 1)
            state = tuple(s.detach() for s in new_state)
            t += 1

        if epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  mse={np.mean(mse_losses):.5f}  "
                  f"g_adv={np.mean(g_adv_losses):.3f}  d={np.mean(d_losses):.3f}  "
                  f"({time.time() - t0:.0f}s)")

    torch.save(G.state_dict(), WEIGHTS_OUT)
    print(f"\nSaved generator to {WEIGHTS_OUT}")
    print("Evaluate it with play.py / compare.py (point them at this weights file).")


if __name__ == "__main__":
    main()