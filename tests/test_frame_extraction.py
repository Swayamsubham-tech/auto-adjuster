import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from frame_extraction import extract_frames  # noqa: E402


def _make_synthetic_video(path, n_frames=30, fps=10, size=(64, 64)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(n_frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        if i < n_frames // 2:
            frame[:, :] = (0, 0, 255)
        else:
            color = ((i * 40) % 255, (i * 80) % 255, (i * 120) % 255)
            frame[:, :] = color
        writer.write(frame)
    writer.release()


def test_extract_frames_produces_output():
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "synthetic.mp4")
        out_dir = os.path.join(tmp, "frames")
        _make_synthetic_video(video_path)
        saved = extract_frames(video_path, out_dir, target_fps=5, hash_threshold=5)
        assert len(saved) > 0
        for p in saved:
            assert os.path.exists(p)


def test_deduplication_reduces_near_identical_frames():
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "synthetic.mp4")
        out_dir = os.path.join(tmp, "frames")
        _make_synthetic_video(video_path, n_frames=30)
        saved_strict = extract_frames(
            video_path, out_dir, target_fps=10, hash_threshold=5
        )
        assert len(saved_strict) < 30
