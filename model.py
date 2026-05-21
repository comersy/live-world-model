"""
dream-mirror / model.py

Un autoencodeur convolutif "prédictif" : il compresse la frame courante
en un petit vecteur latent (l'état du monde percu), puis decode vers la
frame SUIVANTE. C'est ce decalage temporel qui en fait un world model.

Concu pour tourner sur CPU : ~64x64 en niveaux de gris, quelques
centaines de milliers de parametres.
"""

import torch
import torch.nn as nn


class DreamMirror(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- ENCODEUR : [1,64,64] -> latent z [latent_dim] ----
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),   # -> [16,32,32]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # -> [32,16,16]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> [64, 8, 8]
            nn.ReLU(inplace=True),
        )
        self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)  # -> z [latent_dim]

        # ---- DECODEUR : latent z -> prediction de frame_{t+1} [1,64,64] ----
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [32,16,16]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> [16,32,32]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # -> [1,64,64]
            nn.Sigmoid(),  # pixels dans [0,1]
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
        """frame_t [B,1,64,64] -> prediction de frame_{t+1} [B,1,64,64]"""
        z = self.encode(x)
        return self.decode(z)


if __name__ == "__main__":
    # petit test de sanity : verifie les dimensions et compte les parametres
    model = DreamMirror()
    dummy = torch.randn(1, 1, 64, 64)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Entree : {tuple(dummy.shape)}")
    print(f"Sortie : {tuple(out.shape)}")
    print(f"Parametres : {n_params:,}")