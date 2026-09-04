"""
Day 7 — End-to-end Week 1 integration: frame extraction -> SAM segmentation.
"""

import argparse
import os
import json

import numpy as np
import torch
from PIL import Image

from frame_extraction import extract_frames
from sam_segment_demo import load_sam, visualize_masks, masks_to_json


def run_pipeline(video_path, checkpoint, model_type, out_dir, fps, max_frames):
    frames_dir = os.path.join(out_dir, "frames")
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    print("=== Step 1: Frame extraction (Day 4 code) ===")
    frame_paths = extract_frames(video_path, frames_dir, target_fps=fps)
    print(f"Extracted {len(frame_paths)} de-duplicated frames.")

    if len(frame_paths) == 0:
        raise SystemExit("No frames extracted — check the video file and try again.")

    frame_paths = frame_paths[:max_frames]
    print(f"Running SAM on the first {len(frame_paths)} frame(s).")

    print("\n=== Step 2: Zero-shot segmentation (Day 6 code) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_generator = load_sam(checkpoint, model_type, device)

    pipeline_summary = []
    for i, frame_path in enumerate(frame_paths):
        print(f"\n[{i + 1}/{len(frame_paths)}] Processing {frame_path}")
        image_pil = Image.open(frame_path).convert("RGB")
        image_np = np.array(image_pil)

        masks = mask_generator.generate(image_np)
        print(f"  -> {len(masks)} candidate masks found")

        base_name = os.path.splitext(os.path.basename(frame_path))[0]
        overlay_path = os.path.join(masks_dir, f"{base_name}_overlay.png")
        visualize_masks(image_np, masks, overlay_path)

        pipeline_summary.append(
            {
                "frame": frame_path,
                "overlay": overlay_path,
                "num_masks": len(masks),
                "masks": masks_to_json(masks),
            }
        )

    summary_path = os.path.join(out_dir, "week1_pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(pipeline_summary, f, indent=2)

        print("\n=== Pipeline complete ===")
    print(f"Frames:  {frames_dir}")
    print(f"Masks:   {masks_dir}")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"]
    )
    parser.add_argument("--out_dir", default="outputs/day7")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=5)
    args = parser.parse_args()

    run_pipeline(
        args.video,
        args.checkpoint,
        args.model_type,
        args.out_dir,
        args.fps,
        args.max_frames,
    )


if __name__ == "__main__":
    main()
