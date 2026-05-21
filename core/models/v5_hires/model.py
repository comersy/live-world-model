"""
core / models / v5_hires / model.py

ConvLSTM + residual world model, sized for 128x128 grayscale input.

Same design as v4 (recurrent ConvLSTM cell, predicts a residual delta added to
the current frame), but with one extra down/up sampling stage so it can handle
the higher resolution. The bottleneck stays at 8x8 so the recurrent memory
operates on a compact, abstract representation.

Generator-only; trained offline (see offline/train.py) and run live
(see live/live.py).
"""

import torch
import torch.nn as nn

SIZE = 128  # input resolution this model expects


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels + hidden_channels,
                              4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, height, width, device):
        h = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        return h, c


class WorldModel(nn.Module):
    def __init__(self, hidden_channels: int = 64):
        super().__init__()
        self.hidden_channels = hidden_channels

        # encoder: [1,128,128] -> [32,8,8]  (four /2 stages)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # -> [32,64,64]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # -> [32,32,32]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # -> [32,16,16]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # -> [32, 8, 8]
            nn.ReLU(inplace=True),
        )

        self.cell = ConvLSTMCell(in_channels=32, hidden_channels=hidden_channels)

        # decoder: hidden [64,8,8] -> delta [1,128,128]  (four x2 stages)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 32, 4, stride=2, padding=1),  # -> [32,16,16]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),               # -> [32,32,32]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),               # -> [32,64,64]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),                # -> [1,128,128]
            nn.Tanh(),  # delta in [-1,1]
        )

    def init_state(self, batch, device):
        return self.cell.init_state(batch, 8, 8, device)

    def step(self, frame, state):
        feat = self.encoder(frame)
        h, c = self.cell(feat, state)
        delta = self.decoder(h)
        next_pred = (frame + delta).clamp(0, 1)
        return next_pred, (h, c)

    def forward(self, frame, state):
        return self.step(frame, state)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    model = WorldModel()
    state = model.init_state(1, torch.device("cpu"))
    frame = torch.rand(1, 1, SIZE, SIZE)
    for _ in range(3):
        pred, state = model.step(frame, state); frame = pred
    n = sum(p.numel() for p in model.parameters())
    print(f"input {SIZE}x{SIZE} -> output {tuple(pred.shape)}")
    print(f"hidden state {tuple(state[0].shape)}")
    print(f"parameters: {n:,}")