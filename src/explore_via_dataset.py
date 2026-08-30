"""
Day 3 — Exploration for VIA-format (VGG Image Annotator) datasets.
Unlike a folder-per-class dataset, VIA stores everything in one JSON:
each image has a list of "regions" (polygons), and each region has its own
damage label in region_attributes.
"""
import argparse
import json
import os
from collections import defaultdict, Counter

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def load_via_annotations(json_path: str, img_dir: str) -> list:
    with open(json_path) as f:
        data = json.load(f)

    # VIA sometimes nests under "_via_img_metadata", sometimes is flat at top level.
    img_metadata = data.get("_via_img_metadata", data)

    rows = []
    for entry in img_metadata.values():
        filename = entry["filename"]
        img_path = os.path.join(img_dir, filename)
        for region in entry.get("regions", []):
            label = region.get("region_attributes", {}).get("damage", "unlabeled")
            points_x = region["shape_attributes"].get("all_points_x", [])
            points_y = region["shape_attributes"].get("all_points_y", [])
            rows.append({
                "image_path": img_path,
                "damage_label": label,
                "polygon_x": points_x,
                "polygon_y": points_y,
            })
    return rows


def report_stats(rows: list) -> None:
    by_class = Counter(r["damage_label"] for r in rows)
    unique_images = len(set(r["image_path"] for r in rows))

    print(f"\nTotal regions (individual damage annotations): {len(rows)}")
    print(f"Unique images with at least one annotation: {unique_images}")
    print("\n=== Damage class distribution (by region, not image) ===")
    for cls, count in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s} {count:6d}")

    if len(by_class) > 1:
        counts = list(by_class.values())
        ratio = max(counts) / max(min(counts), 1)
        print(f"\nClass imbalance ratio (largest/smallest): {ratio:.1f}x")


def save_sample_grid(rows: list, out_dir: str, per_class: int = 3) -> None:
    """Crops the bounding box of each polygon and shows a few samples per class."""
    os.makedirs(out_dir, exist_ok=True)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["damage_label"]].append(r)

    n_classes = len(by_class)
    fig, axes = plt.subplots(n_classes, per_class, figsize=(per_class * 3, n_classes * 3))
    if n_classes == 1:
        axes = [axes]

    for row_idx, (cls, cls_rows) in enumerate(by_class.items()):
        for col in range(per_class):
            ax = axes[row_idx][col] if n_classes > 1 else axes[col]
            ax.axis("off")
            if col < len(cls_rows):
                r = cls_rows[col]
                try:
                    with Image.open(r["image_path"]) as img:
                        img = img.convert("RGB")
                        xs, ys = r["polygon_x"], r["polygon_y"]
                        if xs and ys:
                            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                            crop = img.crop((x1, y1, x2, y2))
                            ax.imshow(crop)
                        if col == 0:
                            ax.set_title(cls, fontsize=10, loc="left")
                except Exception as e:
                    print(f"  WARNING: could not load {r['image_path']}: {e}")

    out_path = os.path.join(out_dir, "via_sample_grid.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved sample grid to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_dir", default="outputs/day3")
    args = parser.parse_args()

    rows = load_via_annotations(args.json_path, args.img_dir)
    if not rows:
        raise SystemExit("ERROR: no annotated regions found — check the JSON structure.")

    report_stats(rows)
    save_sample_grid(rows, args.out_dir)


if __name__ == "__main__":
    main()
