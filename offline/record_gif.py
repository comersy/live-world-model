"""
offline / record_gif.py

Record a short looping clip from the webcam to use as offline training data.

  D = start/stop recording   Q = quit
Saves data/loop.gif (at the model resolution) and data/loop_big.gif (for viewing).

Run from the project root:   python -m offline.record_gif
"""

import cv2
import numpy as np
import imageio

from core.models.v5_hires.model import SIZE
from core.data_io import frame_to_gray

DISPLAY_SCALE = 4
WEBCAM_INDEX = 0
FPS = 15
OUT_GIF = "data/loop.gif"


def main():
    import os
    os.makedirs("data", exist_ok=True)
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {WEBCAM_INDEX}).")
    print("D = start/stop recording, Q = quit")

    recording = False
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = (frame_to_gray(frame, SIZE) * 255).astype(np.uint8)
        if recording:
            frames.append(small.copy())

        prev = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        prev = cv2.resize(prev, (SIZE * DISPLAY_SCALE, SIZE * DISPLAY_SCALE),
                          interpolation=cv2.INTER_NEAREST)
        if recording:
            cv2.circle(prev, (24, 24), 10, (0, 0, 255), -1)
            cv2.putText(prev, f"REC {len(frames)}", (44, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(prev, "D = record", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("record", prev)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            if not recording:
                frames = []
                recording = True
                print("recording...")
            else:
                recording = False
                print(f"stopped: {len(frames)} frames")
                if len(frames) >= 2:
                    dur = 1.0 / FPS
                    big = [cv2.resize(f, (SIZE * DISPLAY_SCALE, SIZE * DISPLAY_SCALE),
                                      interpolation=cv2.INTER_NEAREST) for f in frames]
                    imageio.mimsave(OUT_GIF,
                                    [cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) for f in big],
                                    duration=dur, loop=0)
                    print(f"saved {OUT_GIF}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()