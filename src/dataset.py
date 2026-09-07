"""
Week 2, Day 3 — PyTorch Dataset for damage crops.
"""

import csv

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(split, image_size=224):
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


class DamageCropDataset(Dataset):
    def __init__(self, manifest_csv, split, image_size=224, class_to_idx=None):
        with open(manifest_csv) as f:
            all_rows = list(csv.DictReader(f))

        self.rows = [r for r in all_rows if r["split"] == split]
        if len(self.rows) == 0:
            raise ValueError(f"No rows found for split='{split}' in {manifest_csv}")

        if class_to_idx is None:
            classes = sorted(set(r["damage_label"] for r in all_rows))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.transform = build_transforms(split, image_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["crop_path"]).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[row["damage_label"]]
        return image, torch.tensor(label, dtype=torch.long)

    def class_counts(self):
        counts = {c: 0 for c in self.class_to_idx}
        for r in self.rows:
            counts[r["damage_label"]] += 1
        return counts
