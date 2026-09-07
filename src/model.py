"""
Week 2, Day 3 — Model architecture: fine-tuned ResNet18.
"""

import torch.nn as nn
from torchvision import models


def build_damage_classifier(num_classes, freeze_backbone=True, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
