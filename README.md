<h1 align="center">live-world-model</h1>

<p align="center">
  <em>A self-supervised world model that learns to predict a live webcam feed, then dreams the future on its own.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/runs%20on-CPU-success.svg" alt="CPU only">
  <img src="https://img.shields.io/badge/dataset-none-lightgrey.svg" alt="No dataset">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
</p>

---

## What this is

A **world model** learns the dynamics of its environment: given what it sees
now, it predicts what comes next. If it learns those dynamics well enough, it
can **dream** — generate the future on its own by feeding its predictions back
into itself, with no input at all.

The goal of this project is to do that **live, on a webcam**: you move in front
of the camera, the model learns to predict your motion in real time, and when
you press a key it keeps the motion going by itself — a small model imagining
what you would do next. Everything runs on **CPU**, from scratch, with no
dataset and no pretraining. The only data the model ever sees is whatever is in
front of the camera.

The display shows three panels side by side:

```
[ REAL ]   [ PREDICTION ]   [ ERROR ]
```

Stay still and the prediction sharpens. Move and the error map lights up where
reality diverged from what the model expected. Press **D** to drop the camera
and watch the model dream.

## Quick start

```bash
pip install -r requirements.txt
python live.py        # live webcam world model; D = dream, Q = quit
```

If the webcam does not open, change `WEBCAM_INDEX` at the top of `live.py`
(try 0, then 1, then 2).

## Project layout

```
live-world-model/
├── live.py            # the main goal: live webcam world model + dream (D toggles)
├── record_gif.py      # record a short looping clip from the webcam -> data/loop.gif
├── train_offline.py   # train a model intensively on a recorded loop (diagnostic bench)
├── play.py            # load a trained model and watch it predict/dream on the loop
├── requirements.txt
├── data/
│   └── loop.gif       # a recorded loop used as a reproducible test signal
└── models/            # one folder per model version (the development history)
    ├── v1_frame_to_frame/
    ├── v2_temporal/
    ├── v3_convlstm/
    └── v4_offline/     # current best: larger ConvLSTM, trained offline with rollout
```

Each `models/<version>/` holds its own `model.py`, a `NOTES.md` describing what
that version changed and why, and (for trained versions) a `weights.pt`.

## The two ways to run it

**1. Live (the goal).** `python live.py` learns online from the webcam, frame by
frame, and dreams when you press D. This is the real target of the project.

**2. Offline bench (diagnostic).** Live training turned out to be hard to debug:
when the dream failed, it was unclear whether the model, the training, or the
setup was at fault. So a controlled test bench was built:

```bash
python record_gif.py     # press D to start/stop; saves data/loop.gif
python train_offline.py  # trains intensively on that loop, saves weights.pt
python play.py           # watch it predict the loop; press D to dream
```

Because the loop is a fixed, repeatable signal, the offline bench makes results
reproducible and lets architectures be compared fairly. It is a means to an end:
its lessons feed back into the live version.

## How it works

The model is a small **ConvLSTM** that carries a hidden state through time. Each
step it takes the current frame plus its memory and predicts the next frame. Two
design choices matter:

- **Residual prediction.** The model predicts the *change* between frames, not
  the whole frame: `next = current + delta`. A static scene means `delta ≈ 0`,
  so the model spends its capacity on where motion actually happens instead of
  re-drawing the static background.

- **Recurrent memory.** The hidden state propagates over time, so the model can
  represent ongoing motion (direction, velocity) rather than guessing from a
  single still frame.

Everything is sized for CPU: 64×64 grayscale frames, a compact network, one
update per frame in live mode.

## The journey: v1 → v4

Each version exists because the previous one hit a concrete, measured wall.

**v1 — frame to frame.** The simplest world model: predict the next frame from a
single current frame. Problem: with no history it cannot perceive motion. A
moving hand passes through the same position going left and going right, and the
model cannot tell which — so it defaults to copying the previous frame.

**v2 — temporal memory.** Feed the last *N* frames stacked together so the model
can see motion direction. Better at predicting movement, but stacking frames is
a blunt way to handle time and it still struggled to capture a full cycle.

**v3 — ConvLSTM + residual.** Replace the frame stack with a recurrent ConvLSTM
(a proper running memory) and predict the residual instead of the full frame.
This fixed the "frozen copy" tendency and learned real motion — but in **dream
mode the motion still faded out** over time.

**v4 — offline + rollout.** Two findings cracked the fading dream:

- *Live training was too slow.* An overfit test proved the model **could** learn
  the loop's dynamics, but only with hundreds of passes — far more than live
  training delivers in a few minutes. Training offline on the recorded loop
  gives it those passes.
- *Exposure bias.* One-step training never teaches the model to run on its own
  outputs, so in closed-loop dreaming its small errors snowball and motion dies.
  The fix is **rollout training**: during training, periodically force the model
  to predict several steps from its own predictions and penalize the whole
  rollout. It learns to stay coherent on its own — exactly what dreaming needs.

v4 is a larger ConvLSTM (~300k params) trained this way. With offline + rollout,
the dream **sustains motion** instead of collapsing.

The open thread: bring these two lessons (enough repetitions, rollout) back into
the **live** setting, so the dream holds in real time — the original goal.

## Settings worth knowing

- `WEBCAM_INDEX` (live.py) — which camera to use.
- `SOURCE` (live.py) — `"webcam"` or `"video"` to run on a looping file instead.
- `EPOCHS`, `ROLLOUT_LEN`, `ROLLOUT_EVERY` (train_offline.py) — training length
  and how aggressively to train closed-loop stability.
- A `metrics.csv` logging block in live.py (clearly delimited) records loss,
  motion, and dream stability for diagnosis; delete the block when not needed.
