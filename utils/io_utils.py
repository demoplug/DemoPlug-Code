"""Frame loading, windowing, and small parsing helpers.

Frames are expected on disk as ``{frames_base}/{camera}/{frame:06d}.png``. The
per-window image list is organized camera-major::

    [cam0_f0, cam0_f1, ..., cam0_fN-1,  cam1_f0, ..., cam1_fN-1,  ...]

so the global view index of camera ``c`` frame ``i`` (within a window of ``n_f``
frames) is ``c * n_f + i``.
"""
from pathlib import Path

import cv2


def frame_path(frames_base, camera, frame):
    return str(Path(frames_base) / camera / f"{frame:06d}.png")


def load_image_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_sliding_windows(frame_start, frame_end, win_size, overlap):
    """Stage 1 windows: overlapping ``[s, e)`` ranges with stride ``win-overlap``."""
    step = win_size - overlap
    windows, s = [], frame_start
    while s < frame_end:
        e = min(s + win_size, frame_end)
        if e - s < overlap + 1:
            break
        windows.append((s, e))
        s += step
    return windows


def build_per_frame_windows(frame_start, frame_end, context):
    """Gaussian-init windows: for each target frame ``f``, a short look-ahead window.

    Returns a list of ``(target_frame, frames_in_window, target_idx)``. The target
    frame sits at index 0 except near the end of the clip, where the window is
    shifted backwards and ``target_idx`` records the target's position.
    """
    out, N = [], context + 1
    for f in range(frame_start, frame_end):
        start, end = f, f + N
        if end > frame_end:
            end = frame_end
            if end - f < 2:  # need >=2 views; pad from behind
                start = max(frame_start, end - N)
        frames = list(range(start, end))
        out.append((f, frames, frames.index(f)))
    return out


def window_image_list(frames_base, cameras, w_start, w_end):
    """Camera-major image path list for one window (see module docstring)."""
    paths = []
    for c in cameras:
        paths.extend(frame_path(frames_base, c, f) for f in range(w_start, w_end))
    return paths


def parse_frames(spec):
    """Parse a frame spec like ``"66-115,120,130-135"`` into a sorted index list."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))
