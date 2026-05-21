"""
core / data_io.py

Shared helpers for loading frames and converting to/from tensors, used by both
offline training and live running so the conversion is identical everywhere.
"""

import cv2
import numpy as np
import torch


def frame_to_gray(frame_bgr, size):
    """BGR webcam frame -> grayscale float array [size,size] in [0,1]"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def load_loop(path, size):
    """load a gif/video as a [T,1,size,size] float tensor in [0,1]"""
    import imageio
    frames = imageio.mimread(path, memtest=False)
    if not frames:
        raise RuntimeError(f"No frames found in {path}")
    gs = []
    for f in frames:
        a = np.asarray(f)
        if a.ndim == 3:
            a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2GRAY)
        small = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
        gs.append(small.astype(np.float32) / 255.0)
    return torch.from_numpy(np.array(gs)).unsqueeze(1)  # [T,1,size,size]


def to_tensor(arr, device):
    """float array [size,size] -> tensor [1,1,size,size]"""
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def to_uint8(arr):
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)