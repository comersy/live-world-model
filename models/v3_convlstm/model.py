"""
live-world-model / models / v3_convlstm / model.py

Two changes vs v2, both aimed at the "frozen dream" failure mode (where the
model just learns to copy the previous frame and the dream stops moving):

1. RECURRENT MEMORY (ConvLSTM)
   Instead of stacking N frames as channels, the model keeps a hidden state
   that propagates through time. It is fed ONE frame per step and accumulates
   memory in its hidden/cell states. This captures longer, cyclic motion far
   better than a fixed stack of frames.

2. RESIDUAL PREDICTION
   The model predicts the CHANGE (delta) between the current frame and the
   next, not the next frame directly. The final prediction is:
       next_frame = current_frame + delta
   If nothing moves, delta = 0, so the lazy "copy" solution is made explicit
   and the network is pushed to spend its capacity on where motion happens.

Designed to run on CPU: 64x64 grayscale, one small ConvLSTM cell.

Version: v3 (convlstm + residual)
  - input: one frame at a time, with a running hidden state
  - output: predicted next frame (current + predicted delta)
  - no action conditioning
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """A single convolutional LSTM cell operating on feature maps."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # one conv produces all four gates (input, forget, output, cell)
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, height, width, device):
        h = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        return h, c


class WorldModel(nn.Module):
    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.hidden_channels = hidden_channels

        # encode a single frame [1,64,64] down to features [16,16,16]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),   # -> [16,32,32]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # -> [16,16,16]
            nn.ReLU(inplace=True),
        )

        # recurrent memory over time, on the 16x16 feature maps
        self.cell = ConvLSTMCell(in_channels=16, hidden_channels=hidden_channels)

        # decode hidden state back to a DELTA image [1,64,64]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 16, kernel_size=4, stride=2, padding=1),  # -> [16,32,32]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),                # -> [1,64,64]
            nn.Tanh(),  # delta in [-1,1]
        )

    def init_state(self, batch, device):
        # feature maps are 16x16 after the encoder
        return self.cell.init_state(batch, 16, 16, device)

    def step(self, frame, state):
        """
        One time step.
          frame: [B,1,64,64] current frame in [0,1]
          state: (h, c) hidden/cell states
        Returns:
          next_pred: [B,1,64,64] predicted next frame in [0,1]
          state:     updated (h, c)
        """
        feat = self.encoder(frame)
        h, c = self.cell(feat, state)
        delta = self.decoder(h)                 # predicted change, in [-1,1]
        next_pred = (frame + delta).clamp(0, 1) # residual prediction
        return next_pred, (h, c)

    def forward(self, frame, state):
        return self.step(frame, state)


if __name__ == "__main__":
    # sanity check: run a few recurrent steps and count parameters
    model = WorldModel()
    state = model.init_state(batch=1, device=torch.device("cpu"))
    frame = torch.rand(1, 1, 64, 64)
    for t in range(3):
        pred, state = model.step(frame, state)
        frame = pred  # feed prediction back (dream-style) just to test shapes
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Per-step output: {tuple(pred.shape)}")
    print(f"Hidden state:    {tuple(state[0].shape)}")
    print(f"Parameters:      {n_params:,}")