"""

Quick standalone recorder to capture a looping clip for use as a reproducible
test signal (and a README demo).

How it works:
  - launch: the webcam preview opens
  - press D : start recording (a red REC dot appears)
  - move / do your gesture
  - press D : stop recording -> a GIF is written to disk
  - press Q : quit

The clip is saved at the same SIZE the model uses (64x64 grayscale by default),
so it can be looped later as the model's input. A scaled-up copy is also saved
for easy viewing / README use.

Run with:   python record_gif.py
"""

import time

import cv2
import numpy as np
import imageio

# ----------------------- settings -----------------------
SIZE = 64             # frame size the model uses (square, grayscale)
DISPLAY_SCALE = 6     # preview magnification
WEBCAM_INDEX = 0      # change to 1, 2... if needed
FPS = 15              # playback fps of the saved GIF
OUT_GIF_BIG = "data/loop.gif"  # scaled-up, nicer to look at
GRAYSCALE = True      # save the loop in grayscale (matches the model input)
# --------------------------------------------------------


def to_gray_small(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam (index {WEBCAM_INDEX}). "
            "Try a different WEBCAM_INDEX, or close other apps using the camera."
        )

    print("Recorder ready.  D = start/stop recording   Q = quit")

    recording = False
    frames = []  # list of small grayscale frames (the loop)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = to_gray_small(frame)  # [SIZE,SIZE] uint8 grayscale

        if recording:
            frames.append(small.copy())

        # build a preview
        preview = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        preview = cv2.resize(
            preview,
            (SIZE * DISPLAY_SCALE, SIZE * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )
        if recording:
            cv2.circle(preview, (24, 24), 10, (0, 0, 255), -1)  # red REC dot
            cv2.putText(preview, f"REC  {len(frames)} frames", (44, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(preview, "D = record", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("record_gif", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            if not recording:
                frames = []
                recording = True
                print("--> recording started")
            else:
                recording = False
                print(f"--> recording stopped: {len(frames)} frames")
                if len(frames) >= 2:
                    save_gifs(frames)
                else:
                    print("    too few frames, nothing saved")

    cap.release()
    cv2.destroyAllWindows()


def save_gifs(frames):
    duration = 1.0 / FPS
    
    # scaled-up version for viewing
    big = [
        cv2.resize(f, (SIZE * DISPLAY_SCALE, SIZE * DISPLAY_SCALE),
                   interpolation=cv2.INTER_NEAREST)
        for f in frames
    ]
    big_rgb = [cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) for f in big]
    imageio.mimsave(OUT_GIF_BIG, big_rgb, duration=duration, loop=0)
    print(f"    saved {OUT_GIF_BIG} (scaled up for viewing)")


if __name__ == "__main__":
    main()