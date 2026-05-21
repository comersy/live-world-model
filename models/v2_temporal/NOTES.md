# v2 — temporal

Adds temporal memory. Instead of predicting from a single frame, the model
predicts the next frame from the last N frames stacked as input channels.

## Characteristics
- **Input:** the last N frames stacked, 64x64 grayscale (default N=4)
- **Output:** prediction of the next frame, 64x64
- **Architecture:** convolutional autoencoder, first conv reads N channels,
  128-dim latent, ~1.14M parameters
- **Memory:** yes — a rolling window of N frames
- **Action conditioning:** none
- **Loss:** MSE

## What it improves over v1
- Can perceive motion. With several frames, the model sees velocity and
  direction, so it can anticipate where moving regions are heading instead of
  defaulting to "next frame == last frame".
- Watch the ERROR column when you move: it should lag less than v1 did.

## Knobs
- `N_FRAMES` in `live.py` sets the memory length. More frames = more motion
  context but slightly heavier. 4 is a good CPU default.

## Files
- `model.py` — the network (takes N input channels)