"""
Day 3 — Exploration for CarDD's FiftyOne-exported format.
"""
import argparse
import json
import os
from collections import defaultdict, Counter

from PIL import Image
import matplotlib.pyplot as plt

LABEL_MAP = {
    "dent": "dent", "scratch": "scratch", "crack": "crack",
    "glass shatter": "shattered_glass", "lamp broken": "broken_lamp", "tire flat": "flat_tire",
}


def resolve_image_path(filepath: str, data_root: str) -> str:
    if os.path.exists(filepath):
        return filepath
    candidate = os.path.join(data_root, filepath)
    if os.path.exists(candidate):
        return candidate
    candidate2 = os.path.join(data_root, "data", os.path.basename(filepath))
    if os.path.exists(candidate2):
        return candidate2
    return None


def load_cardd_fiftyone(samples_json: str, data_root: str) -> list:
    with open(samples_json) as f:
        data = json.load(f)

    rows = []
    n_missing = 0

    for sample in data["samples"]:
        img_path = resolve_image_path(sample["filepath"], data_root)
        if img_path is None:
            n_missing += 1
            continue

        width = height = None
        if sample.get("metadata"):
            width = sample["metadata"].get("width")
            height = sample["metadata"].get("height")

        detections = (sample.get("detections") or {}).get("detections", [])
        for det in detections:
            label = LABEL_MAP.get(det["label"], det["label"])
            nx, ny, nw, nh = det["bounding_box"]
            rows.append({
                "image_path": img_path,
                "damage_label": label,
                "norm_bbox": (nx, ny, nw, nh),
                "img_width": width, "img_height": height,
            })

    if n_missing:
        print(f"WARNING: {n_missing} samples had unresolvable image paths — skipped.")

    return rows


def report_stats(rows: list) -> None:
    by_class = Counter(r["damage_label"] for r in rows)
    unique_images = len(set(r["image_path"] for r in rows))

    print(f"\nTotal detections: {len(rows)}")
    print(f"Unique images with at least one detection: {unique_images}")
    print("\n=== Damage class distribution ===")
    for cls, count in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s} {count:6d}")

    if len(by_class) > 1:
        counts = list(by_class.values())
        ratio = max(counts) / max(min(counts), 1)
        print(f"\nClass imbalance ratio (largest/smallest): {ratio:.1f}x")


def save_sample_grid(rows: list, out_dir: str, per_class: int = 3) -> None:
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
                        w, h = img.size
                        nx, ny, nw, nh = r["norm_bbox"]
                        x1, y1 = int(nx * w), int(ny * h)
                        x2, y2 = int((nx + nw) * w), int((ny + nh) * h)
                        crop = img.crop((x1, y1, x2, y2))
                        ax.imshow(crop)
                        if col == 0:
                            ax.set_title(cls, fontsize=10, loc="left")
                except Exception as e:
                    print(f"  WARNING: could not load {r['image_path']}: {e}")

    out_path = os.path.join(out_dir, "cardd_sample_grid.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved sample grid to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_json", default="data/cardd/samples.json")
    parser.add_argument("--data_root", default="data/cardd")
    parser.add_argument("--out_dir", default="outputs/day3_cardd")
    args = parser.parse_args()

    rows = load_cardd_fiftyone(args.samples_json, args.data_root)
    if not rows:
        raise SystemExit("ERROR: no detections loaded — check samples.json structure.")

    report_stats(rows)
    save_sample_grid(rows, args.out_dir)


if __name__ == "__main__":
    main()
