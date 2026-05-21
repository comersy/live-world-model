# v3 — convlstm + residual

Built specifically to fix the **frozen dream** seen in v2 (where the model
learns to copy the previous frame, so in closed loop the image stops moving).

## Two changes vs v2

1. **Recurrent memory (ConvLSTM).** Instead of stacking N frames as channels,
   the model carries a hidden state through time, fed one frame per step. This
   captures longer cyclic motion than a fixed frame stack.

2. **Residual prediction.** The model outputs the *change* (delta) between the
   current and next frame; the prediction is `current + delta`. A static scene
   means delta = 0, so the lazy copy solution is explicit and the network is
   pushed to model where motion actually happens.

## Characteristics
- **Input:** one 64x64 grayscale frame per step + carried hidden state
- **Output:** predicted next frame = current + predicted delta
- **Memory:** recurrent hidden/cell state (32 channels, 16x16)
- **Parameters:** ~68k (lighter than v2)
- **Loss:** MSE on the predicted next frame

## Training detail
`live.py` uses truncated backprop through time (TBPTT): it accumulates loss
over a short window (`TBPTT` steps) then steps the optimizer and detaches the
state. This keeps recurrent live training cheap and stable on CPU.

## How to judge it
Do a regular, repetitive motion (e.g. hand sweeping left-right) for a few
minutes, then press D. Success = the dream keeps doing the motion on its own
for a while before drifting. A frozen image means it still hasn't learned the
dynamics: train longer, with a more regular motion.

## Files
- `model.py` — ConvLSTM cell + residual world model