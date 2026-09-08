"""
Week 2, Day 5 — Class imbalance handling via weighted sampling.
"""

import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(dataset):
    """Per-sample weight = 1 / (count of that sample's class). Rare classes get sampled more."""
    class_counts = dataset.class_counts()
    sample_weights = [1.0 / class_counts[row["damage_label"]] for row in dataset.rows]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def compute_class_weights_for_loss(dataset):
    """
    Alternative/complementary: weight the LOSS instead of (or with) the sampler.
    Usually pick ONE approach, not both — combining can over-correct.
    """
    class_counts = dataset.class_counts()
    idx_to_class = dataset.idx_to_class
    weights = [
        (
            1.0 / class_counts[idx_to_class[i]]
            if class_counts[idx_to_class[i]] > 0
            else 0.0
        )
        for i in range(len(idx_to_class))
    ]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor / weights_tensor.mean()
