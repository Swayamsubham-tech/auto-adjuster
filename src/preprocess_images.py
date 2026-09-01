"""
Day 5 — Preprocessing: normalize images before feeding them to SAM.
"""

import argparse
import os

from PIL import Image, ImageOps
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resize_keep_aspect(img, max_dim):
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def process_directory(in_dir, out_dir, max_dim):
    os.makedirs(out_dir, exist_ok=True)
    all_files = []
    for root, _, files in os.walk(in_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                all_files.append(os.path.join(root, f))

    print(f"Found {len(all_files)} candidate images under {in_dir}")
    n_ok, n_failed, failed_files = 0, 0, []

    for path in tqdm(all_files, desc="Preprocessing"):
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                img = resize_keep_aspect(img, max_dim)

                rel_path = os.path.relpath(path, in_dir)
                out_path = os.path.join(out_dir, rel_path)
                out_path = os.path.splitext(out_path)[0] + ".jpg"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                img.save(out_path, "JPEG", quality=92)
                n_ok += 1
        except Exception as e:
            n_failed += 1
            failed_files.append((path, str(e)))

    print(f"\nProcessed OK: {n_ok}")
    print(f"Failed:       {n_failed}")
    if failed_files:
        print("First few failures:")
        for path, err in failed_files[:10]:
            print(f"  - {path}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--max_dim", type=int, default=1024)
    args = parser.parse_args()
    process_directory(args.in_dir, args.out_dir, args.max_dim)


if __name__ == "__main__":
    main()
