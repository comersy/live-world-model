"""
live-world-model / models / v2_temporal / model.py

A predictive convolutional autoencoder WITH temporal memory.

Instead of predicting frame_{t+1} from a single frame, it predicts from the
last N frames stacked as input channels. Seeing several frames at once lets
the model perceive motion (velocity and direction), so it can anticipate
where things are heading instead of being stuck at "next frame == last frame".

Designed to run on CPU: 64x64 grayscale, N stacked frames.

Version: v2 (temporal)
  - input: the last N frames stacked as channels  [N, 64, 64]
  - output: prediction of the single next frame   [1, 64, 64]
  - no action conditioning (that is v3)
"""

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(self, n_frames: int = 4, latent_dim: int = 128):
        super().__init__()
        self.n_frames = n_frames          # how many past frames feed the model
        self.latent_dim = latent_dim

        # ---- ENCODER: [N,64,64] -> latent z [latent_dim] ----
        # only the first conv changes vs v1: it now reads N input channels
        self.encoder = nn.Sequential(
            nn.Conv2d(n_frames, 16, kernel_size=4, stride=2, padding=1),  # -> [16,32,32]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),        # -> [32,16,16]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),        # -> [64, 8, 8]
            nn.ReLU(inplace=True),
        )
        self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)

        # ---- DECODER: latent z -> prediction of the next frame [1,64,64] ----
        # output stays a single frame (the next one)
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [32,16,16]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> [16,32,32]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # -> [1,64,64]
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(-1, 64, 8, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """stacked frames [B,N,64,64] -> prediction of next frame [B,1,64,64]"""
        z = self.encode(x)
        return self.decode(z)


if __name__ == "__main__":
    # sanity check: verify dimensions and count parameters
    n = 4
    model = WorldModel(n_frames=n)
    dummy = torch.randn(1, n, 64, 64)  # a stack of N frames
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {tuple(dummy.shape)}  (a stack of {n} frames)")
    print(f"Output: {tuple(out.shape)}  (the predicted next frame)")
    print(f"Parameters: {n_params:,}")