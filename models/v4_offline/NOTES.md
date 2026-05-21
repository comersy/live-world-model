# v5 — offline

The version that finally makes the dream hold together. Two ideas, both proven
on the recorded loop:

## 1. Train OFFLINE, not live
Live training only shows the model each part of the motion a few times per
minute — not enough to learn the dynamics. An overfit test proved the model
*can* learn the loop, but needs hundreds of passes. `train_offline.py` does
those passes in ~2 minutes on the recorded `data/loop.gif`.

## 2. Rollout training (the real fix for the collapsing dream)
Plain one-step training never teaches the model to run on its own outputs, so
in dream mode (closed loop) its errors snowball and motion dies. `train_offline.py`
periodically forces a closed-loop rollout (predict ROLLOUT_LEN steps from its
own predictions) and penalizes the whole rollout. This is the standard fix for
exposure bias.

## Bigger model
The 68k-param v3 plateaued (residual error on motion never reached zero). v5 is
larger (~300k params): deeper encoder/decoder, 8x8 bottleneck, 64-channel
ConvLSTM hidden state. More capacity to fit the motion precisely.

## Measured result
After offline + rollout training, dream motion is *sustained* over 120+ steps
(survival ratio ~1.0) instead of collapsing to ~0 (ratio ~0.05 in the live
sessions). The dream keeps moving on its own.

## How to use
1. Record a loop with `record_gif.py` -> `data/loop.gif`
2. `python train_offline.py`  (saves weights.pt here)
3. Run live.py pointed at v5 with weights loaded, press D to dream

## Files
- `model.py`   — larger ConvLSTM + residual
- `weights.pt` — trained weights (created by train_offline.py)