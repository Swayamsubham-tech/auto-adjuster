"""
Day 6 — SAM zero-shot segmentation demo.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


def load_sam(checkpoint, model_type, device):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}.")

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=200,
    )
    return mask_generator


def visualize_masks(image, masks, out_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    if len(masks) == 0:
        plt.title("No masks found")
    else:
        sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)
        overlay = np.ones((image.shape[0], image.shape[1], 4))
        overlay[:, :, 3] = 0
        rng = np.random.default_rng(42)
        for m in sorted_masks:
            color_mask = np.concatenate([rng.random(3), [0.45]])
            overlay[m["segmentation"]] = color_mask
        plt.imshow(overlay)
        plt.title(f"{len(masks)} candidate regions found (SAM zero-shot)")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def masks_to_json(masks):
    summary = []
    for i, m in enumerate(masks):
        summary.append({
            "mask_id": i,
            "bbox_xywh": [int(v) for v in m["bbox"]],
            "area_px": int(m["area"]),
            "predicted_iou": float(m["predicted_iou"]),
            "stability_score": float(m["stability_score"]),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--out_dir", default="outputs/day6")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("NOTE: CPU run will be slow. Expected only if GPU isn't detected.")

    os.makedirs(args.out_dir, exist_ok=True)
    mask_generator = load_sam(args.checkpoint, args.model_type, device)

    image_pil = Image.open(args.image).convert("RGB")
    image_np = np.array(image_pil)

    print(f"Running automatic mask generation on {args.image} ...")
    masks = mask_generator.generate(image_np)
    print(f"Found {len(masks)} candidate masks.")

    base_name = os.path.splitext(os.path.basename(args.image))[0]
    overlay_path = os.path.join(args.out_dir, f"{base_name}_masks_overlay.png")
    visualize_masks(image_np, masks, overlay_path)
    print(f"Saved visualization: {overlay_path}")

    json_path = os.path.join(args.out_dir, f"{base_name}_masks.json")
    with open(json_path, "w") as f:
        json.dump(masks_to_json(masks), f, indent=2)
    print(f"Saved mask metadata: {json_path}")


if __name__ == "__main__":
    main()
