"""
GAN setup for sharper, moving dreams.

- GENERATOR: the existing ConvLSTM + residual world model (re-exported,
  unchanged). It predicts the next frame.

- DISCRIMINATOR: a small CNN that looks at a PAIR of frames
  (previous_frame, next_frame) stacked as 2 channels, and judges whether the
  transition is real (from the loop) or fake (produced by the generator).
  Judging the pair (not a single frame) is what pushes the generator to make
  realistic MOTION, not just a realistic still image.

The discriminator is only used during training (train_gan.py) and thrown away
afterwards — play.py / dream only need the generator.

"""

import torch
import torch.nn as nn

# generator = the offline world model (ConvLSTM + residual)
from models.v4_offline.model import WorldModel  # adapt to v4_offline in your repo

__all__ = ["WorldModel", "Discriminator"]


class Discriminator(nn.Module):
    """
    Looks at a (prev_frame, next_frame) pair -> probability the pair is REAL.
    Input:  [B, 2, 64, 64]   (two stacked grayscale frames)
    Output: [B, 1]           (logit; use with BCEWithLogitsLoss)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 4, stride=2, padding=1),    # -> [32,32,32]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # -> [64,16,16]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),   # -> [64, 8, 8]
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Linear(64 * 8 * 8, 1)

    def forward(self, prev_frame, next_frame):
        x = torch.cat([prev_frame, next_frame], dim=1)  # [B,2,64,64]
        h = self.net(x).flatten(1)
        return self.head(h)  # logit


if __name__ == "__main__":
    g = WorldModel()
    d = Discriminator()
    state = g.init_state(1, torch.device("cpu"))
    frame = torch.rand(1, 1, 64, 64)
    pred, state = g.step(frame, state)
    logit = d(frame, pred)
    gp = sum(p.numel() for p in g.parameters())
    dp = sum(p.numel() for p in d.parameters())
    print(f"generator pred: {tuple(pred.shape)}  params: {gp:,}")
    print(f"discriminator logit: {tuple(logit.shape)}  params: {dp:,}")