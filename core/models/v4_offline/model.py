"""

Same ConvLSTM + residual design as v3, but LARGER. The overfit test showed the
68k-param model could learn the loop's dynamics but plateaued (residual error
on motion never reached zero). v5 increases capacity so it can fit the loop
more precisely, and is meant to be trained OFFLINE on a recorded loop with many
epochs (see train_offline.py) rather than live frame-by-frame.

Changes vs v3:
  - deeper encoder/decoder (extra conv stage, more channels)
  - bigger ConvLSTM hidden state
  - bottleneck at 8x8 instead of 16x16 (more abstraction, more context)

Version: v4 (offline-trained, larger ConvLSTM + residual)
"""

import torch
import torch.nn as nn


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

        # encode a frame [1,64,64] down to features [32,8,8]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # -> [32,32,32]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # -> [32,16,16]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # -> [32, 8, 8]
            nn.ReLU(inplace=True),
        )

        self.cell = ConvLSTMCell(in_channels=32, hidden_channels=hidden_channels)

        # decode hidden state back to a DELTA image [1,64,64]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 32, 4, stride=2, padding=1),  # -> [32,16,16]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),               # -> [32,32,32]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),                # -> [1,64,64]
            nn.Tanh(),  # delta in [-1,1]
        )

    def init_state(self, batch, device):
        return self.cell.init_state(batch, 8, 8, device)  # features are 8x8

    def step(self, frame, state):
        feat = self.encoder(frame)
        h, c = self.cell(feat, state)
        delta = self.decoder(h)
        next_pred = (frame + delta).clamp(0, 1)
        return next_pred, (h, c)

    def forward(self, frame, state):
        return self.step(frame, state)


if __name__ == "__main__":
    model = WorldModel()
    state = model.init_state(1, torch.device("cpu"))
    frame = torch.rand(1, 1, 64, 64)
    for _ in range(3):
        pred, state = model.step(frame, state); frame = pred
    n = sum(p.numel() for p in model.parameters())
    print(f"Per-step output: {tuple(pred.shape)}")
    print(f"Hidden state:    {tuple(state[0].shape)}")
    print(f"Parameters:      {n:,}")