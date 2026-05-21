"""
live-world-model / models / v1_frame_to_frame / model.py

A predictive convolutional autoencoder: it compresses the current frame
into a small latent vector (the perceived "state of the world"), then
decodes it into the NEXT frame. That temporal offset is what makes this a
world model rather than a plain autoencoder.

Designed to run on CPU: 64x64 grayscale, ~1.1M parameters.

Version: v1 (frame-to-frame)
  - predicts frame_{t+1} from a SINGLE frame_t
  - no temporal memory, no action conditioning
"""

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- ENCODER: [1,64,64] -> latent z [latent_dim] ----
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),   # -> [16,32,32]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # -> [32,16,16]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> [64, 8, 8]
            nn.ReLU(inplace=True),
        )
        self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)  # -> z [latent_dim]

        # ---- DECODER: latent z -> prediction of frame_{t+1} [1,64,64] ----
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [32,16,16]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> [16,32,32]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # -> [1,64,64]
            nn.Sigmoid(),  # pixels in [0,1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        z = self.fc_enc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(-1, 64, 8, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """frame_t [B,1,64,64] -> prediction of frame_{t+1} [B,1,64,64]"""
        z = self.encode(x)
        return self.decode(z)


if __name__ == "__main__":
    # sanity check: verify dimensions and count parameters
    model = WorldModel()
    dummy = torch.randn(1, 1, 64, 64)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Parameters: {n_params:,}")