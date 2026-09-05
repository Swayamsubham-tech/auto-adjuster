"""
Week 2, Day 1 — Dataset preparation.
Unifies CarDD (FiftyOne export format) and the Kaggle VIA dataset into one manifest.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

CARDD_LABEL_MAP = {
    "dent": "dent",
    "scratch": "scratch",
    "crack": "crack",
    "glass shatter": "shattered_glass",
    "lamp broken": "broken_lamp",
    "tire flat": "flat_tire",
}

VIA_LABEL_MAP = {
    "Dent": "dent",
    "Scratch": "scratch",
    "Shatter": "shattered_glass",
    "Dislocation": "dislocation",
}


def resolve_image_path(filepath, data_root):
    if os.path.exists(filepath):
        return filepath
    candidate = os.path.join(data_root, filepath)
    if os.path.exists(candidate):
        return candidate
    candidate2 = os.path.join(data_root, "data", os.path.basename(filepath))
    if os.path.exists(candidate2):
        return candidate2
    return None


def load_cardd(samples_json, data_root):
    """Parses CarDD's FiftyOne-format samples.json (normalized bboxes)."""
    if not samples_json or not os.path.exists(samples_json):
        print(f"WARNING: {samples_json} not found — skipping CarDD.")
        return []

    with open(samples_json) as f:
        data = json.load(f)

    rows = []
    n_missing = 0
    for sample in data["samples"]:
        img_path = resolve_image_path(sample["filepath"], data_root)
        if img_path is None:
            n_missing += 1
            continue

        detections = (sample.get("detections") or {}).get("detections", [])
        for det in detections:
            label = CARDD_LABEL_MAP.get(det["label"], det["label"])
            nx, ny, nw, nh = det["bounding_box"]  # normalized [x, y, w, h]
            rows.append(
                {
                    "image_path": img_path,
                    "source": "cardd",
                    "damage_label": label,
                    "norm_bbox_x": nx,
                    "norm_bbox_y": ny,
                    "norm_bbox_w": nw,
                    "norm_bbox_h": nh,
                    "is_normalized": True,
                }
            )

    if n_missing:
        print(
            f"WARNING: {n_missing} CarDD samples had unresolvable image paths — skipped."
        )
    print(f"CarDD: parsed {len(rows)} detections from {len(data['samples'])} samples.")
    return rows


def load_via_dataset(via_json, img_dir):
    """Parses the Kaggle VIA-format annotations (pixel-space polygon points)."""
    if not via_json or not os.path.exists(via_json):
        print(f"WARNING: {via_json} not found — skipping VIA dataset.")
        return []

    with open(via_json) as f:
        data = json.load(f)

    img_metadata = data.get("_via_img_metadata", data)
    rows = []
    for entry in img_metadata.values():
        filename = entry["filename"]
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            continue

        for region in entry.get("regions", []):
            raw_label = region.get("region_attributes", {}).get("damage", "unlabeled")
            label = VIA_LABEL_MAP.get(raw_label, raw_label.lower())
            points_x = region["shape_attributes"].get("all_points_x", [])
            points_y = region["shape_attributes"].get("all_points_y", [])
            if not points_x or not points_y:
                continue

            x1, y1, x2, y2 = min(points_x), min(points_y), max(points_x), max(points_y)
            rows.append(
                {
                    "image_path": img_path,
                    "source": "kaggle_via",
                    "damage_label": label,
                    "pixel_bbox_x": x1,
                    "pixel_bbox_y": y1,
                    "pixel_bbox_w": x2 - x1,
                    "pixel_bbox_h": y2 - y1,
                    "is_normalized": False,
                }
            )

    print(f"VIA dataset: parsed {len(rows)} detections from {img_dir}.")
    return rows


def assign_splits(rows, val_split, test_split, seed=42):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["damage_label"]].append(r)

    rng = random.Random(seed)
    for cls, cls_rows in by_class.items():
        rng.shuffle(cls_rows)
        n = len(cls_rows)
        n_val = max(1, int(n * val_split)) if n > 2 else 0
        n_test = max(1, int(n * test_split)) if n > 2 else 0
        for i, r in enumerate(cls_rows):
            if i < n_val:
                r["split"] = "val"
            elif i < n_val + n_test:
                r["split"] = "test"
            else:
                r["split"] = "train"
    return rows


def write_manifest(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [
        "image_path",
        "source",
        "damage_label",
        "is_normalized",
        "norm_bbox_x",
        "norm_bbox_y",
        "norm_bbox_w",
        "norm_bbox_h",
        "pixel_bbox_x",
        "pixel_bbox_y",
        "pixel_bbox_w",
        "pixel_bbox_h",
        "split",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Wrote {len(rows)} rows to {out_csv}")


def print_summary(rows):
    by_class, by_split = defaultdict(int), defaultdict(int)
    for r in rows:
        by_class[r["damage_label"]] += 1
        by_split[r["split"]] += 1

    print("\n=== Class distribution ===")
    for cls, count in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s} {count:6d}")

    print("\n=== Split distribution ===")
    for split, count in by_split.items():
        print(f"  {split:10s} {count:6d}")

    if len(by_class) > 1:
        counts = list(by_class.values())
        ratio = max(counts) / max(min(counts), 1)
        print(f"\nClass imbalance ratio (largest/smallest): {ratio:.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cardd_samples_json", default="data/cardd/samples.json")
    parser.add_argument("--cardd_data_root", default="data/cardd")
    parser.add_argument(
        "--via_json", default="data/kaggle_car_damage/train/via_project.json"
    )
    parser.add_argument("--via_img_dir", default="data/kaggle_car_damage/train")
    parser.add_argument("--out_csv", default="data/manifest.csv")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    args = parser.parse_args()

    all_rows = []
    all_rows.extend(load_cardd(args.cardd_samples_json, args.cardd_data_root))
    all_rows.extend(load_via_dataset(args.via_json, args.via_img_dir))

    if not all_rows:
        raise SystemExit("ERROR: no data loaded. Check dataset paths.")

    all_rows = assign_splits(all_rows, args.val_split, args.test_split)
    print_summary(all_rows)
    write_manifest(all_rows, args.out_csv)


if __name__ == "__main__":
    main()
