"""
Unit test for src/preprocess_images.py — verifies resizing and RGB conversion
work correctly on synthetic images, without needing the real dataset present.
"""

import os
import sys
import tempfile

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess_images import process_directory  # noqa: E402


def test_oversized_image_gets_resized():
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)

        Image.new("RGB", (2000, 1500), (255, 0, 0)).save(
            os.path.join(in_dir, "big.jpg")
        )

        process_directory(in_dir, out_dir, max_dim=1024)

        out_path = os.path.join(out_dir, "big.jpg")
        assert os.path.exists(out_path)
        with Image.open(out_path) as img:
            assert max(img.size) <= 1024


def test_rgba_image_gets_converted_to_rgb():
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)

        Image.new("RGBA", (500, 500), (0, 255, 0, 255)).save(
            os.path.join(in_dir, "transparent.png")
        )

        process_directory(in_dir, out_dir, max_dim=1024)

        out_path = os.path.join(out_dir, "transparent.jpg")
        assert os.path.exists(out_path)
        with Image.open(out_path) as img:
            assert img.mode == "RGB"


def test_small_image_is_not_upscaled():
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)

        Image.new("RGB", (100, 100), (0, 0, 255)).save(
            os.path.join(in_dir, "small.jpg")
        )

        process_directory(in_dir, out_dir, max_dim=1024)

        out_path = os.path.join(out_dir, "small.jpg")
        with Image.open(out_path) as img:
            assert img.size == (100, 100)  # unchanged, not upscaled
