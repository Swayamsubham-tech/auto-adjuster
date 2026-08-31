"""
Day 4 — Frame extraction utility. Samples frames from video at a target FPS
and drops near-duplicate frames via perceptual hashing.
"""

import argparse
import os

import cv2
import imagehash
from PIL import Image


def extract_frames(video_path, out_dir, target_fps=3.0, hash_threshold=5):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open {video_path}. Try: "
            f"ffmpeg -i {video_path} -c:v libx264 fixed.mp4"
        )

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(source_fps / target_fps)), 1)

    saved_paths = []
    last_hash = None
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            current_hash = imagehash.phash(pil_frame)

            is_duplicate = (
                last_hash is not None and (current_hash - last_hash) <= hash_threshold
            )

            if not is_duplicate:
                out_path = os.path.join(out_dir, f"frame_{saved_idx:04d}.jpg")
                pil_frame.save(out_path, quality=95)
                saved_paths.append(out_path)
                last_hash = current_hash
                saved_idx += 1

        frame_idx += 1

    cap.release()
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", default="outputs/day4/frames")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--hash_threshold", type=int, default=5)
    args = parser.parse_args()

    saved = extract_frames(args.video, args.out_dir, args.fps, args.hash_threshold)
    print(f"Extracted {len(saved)} de-duplicated frames to {args.out_dir}")
    for p in saved[:5]:
        print(f"  - {p}")
    if len(saved) == 0:
        print("WARNING: zero frames saved — check the video file decoded correctly.")


if __name__ == "__main__":
    main()
