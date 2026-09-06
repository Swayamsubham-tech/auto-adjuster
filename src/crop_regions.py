"""
Week 2, Day 2 — Crop extraction.
Handles BOTH coordinate systems present in the manifest: CarDD's normalized
(0-1) bboxes and the VIA dataset's pixel-space bboxes.
"""

import argparse
import csv
import os

from PIL import Image
from tqdm import tqdm


def resolve_pixel_bbox(row, img_w, img_h, margin):
    """
    Returns (x1, y1, x2, y2) in real pixel coordinates, with margin applied,
    regardless of which coordinate system the row originally used.
    """
    is_normalized = row["is_normalized"] == "True"

    if is_normalized:
        nx = float(row["norm_bbox_x"])
        ny = float(row["norm_bbox_y"])
        nw = float(row["norm_bbox_w"])
        nh = float(row["norm_bbox_h"])
        bbox_x = nx * img_w
        bbox_y = ny * img_h
        bbox_w = nw * img_w
        bbox_h = nh * img_h
    else:
        bbox_x = float(row["pixel_bbox_x"])
        bbox_y = float(row["pixel_bbox_y"])
        bbox_w = float(row["pixel_bbox_w"])
        bbox_h = float(row["pixel_bbox_h"])

    pad_w = bbox_w * margin
    pad_h = bbox_h * margin

    x1 = max(0, bbox_x - pad_w)
    y1 = max(0, bbox_y - pad_h)
    x2 = min(img_w, bbox_x + bbox_w + pad_w)
    y2 = min(img_h, bbox_y + bbox_h + pad_h)

    return int(x1), int(y1), int(x2), int(y2)


def process_manifest(manifest_path, out_dir, margin):
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} manifest rows.")
    n_ok, n_failed = 0, 0
    failed_examples = []
    crop_manifest_rows = []

    for i, row in enumerate(tqdm(rows, desc="Cropping")):
        try:
            with Image.open(row["image_path"]) as img:
                img = img.convert("RGB")
                w, h = img.size

                x1, y1, x2, y2 = resolve_pixel_bbox(row, w, h, margin)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Degenerate crop box: {(x1, y1, x2, y2)}")

                crop = img.crop((x1, y1, x2, y2))

                label = row["damage_label"]
                split = row["split"]
                class_dir = os.path.join(out_dir, split, label)
                os.makedirs(class_dir, exist_ok=True)

                out_path = os.path.join(class_dir, f"{row['source']}_{i:06d}.jpg")
                crop.save(out_path, "JPEG", quality=92)

                crop_manifest_rows.append(
                    {
                        "crop_path": out_path,
                        "damage_label": label,
                        "split": split,
                    }
                )
                n_ok += 1
        except Exception as e:
            n_failed += 1
            if len(failed_examples) < 10:
                failed_examples.append((row.get("image_path"), str(e)))

    print(f"\nCropped OK: {n_ok}")
    print(f"Failed:     {n_failed}")
    if failed_examples:
        print("First few failures:")
        for path, err in failed_examples:
            print(f"  - {path}: {err}")

    crop_csv = os.path.join(out_dir, "crop_manifest.csv")
    with open(crop_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "damage_label", "split"])
        writer.writeheader()
        writer.writerows(crop_manifest_rows)
    print(f"Wrote crop manifest: {crop_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", default="data/crops")
    parser.add_argument("--margin", type=float, default=0.15)
    args = parser.parse_args()

    process_manifest(args.manifest, args.out_dir, args.margin)


if __name__ == "__main__":
    main()
