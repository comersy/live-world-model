# v1 — frame-to-frame

The first model. The simplest possible world model: it predicts the next
frame from a single current frame.

## Characteristics
- **Input:** one grayscale frame, 64x64
- **Output:** prediction of the next frame, 64x64
- **Architecture:** convolutional autoencoder, 128-dim latent, ~1.1M parameters
- **Memory:** none (each prediction uses only the latest frame)
- **Action conditioning:** none
- **Loss:** MSE

## Known limitations
- Cannot anticipate motion: with no history, it cannot tell which way a moving
  object is heading. Predictions of moving regions are blurry.
- This is the motivation for v2 (temporal memory).

## Files
- `model.py` — the network definition
- `weights.pt` — saved weights (created automatically on first run)