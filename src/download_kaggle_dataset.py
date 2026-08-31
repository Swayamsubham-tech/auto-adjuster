"""
Day 3 — Dataset download utility.
Downloads a public Kaggle vehicle-damage dataset.
"""

import argparse
import os
import sys
import zipfile


def check_kaggle_credentials() -> None:
    cred_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(cred_path):
        sys.exit(
            "ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json.\n"
            "Create one at kaggle.com > Account > API > Create Legacy API Key."
        )


def download_and_extract(dataset_slug: str, dest_dir: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    os.makedirs(dest_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading '{dataset_slug}' to {dest_dir} ...")
    api.dataset_download_files(dataset_slug, path=dest_dir, quiet=False)

    zip_name = dataset_slug.split("/")[-1] + ".zip"
    zip_path = os.path.join(dest_dir, zip_name)

    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)
        print("Done. Extracted contents:")
        for name in sorted(os.listdir(dest_dir))[:20]:
            print(f"  - {name}")
    else:
        print("WARNING: expected zip not found — check dataset slug/credentials.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, help="e.g. 'anujms/car-damage-detection'"
    )
    parser.add_argument("--dest", default="data/kaggle_car_damage")
    args = parser.parse_args()

    check_kaggle_credentials()
    download_and_extract(args.dataset, args.dest)


if __name__ == "__main__":
    main()
