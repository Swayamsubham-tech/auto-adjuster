"""
Week 3, Day 2 — Full pipeline: video -> Damage JSON.
"""

import argparse
import json
import os
import uuid

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from fusion import run_fusion_pipeline
from confidence import apply_confidence_thresholds
from frame_extraction import extract_frames
from sam_segment_demo import load_sam
from model import build_damage_classifier

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASSIFIER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_classifier(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_damage_classifier(
        num_classes=len(class_to_idx), freeze_backbone=True, pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, idx_to_class


@torch.no_grad()
def classify_crop(model, idx_to_class, crop_image, device):
    tensor = CLASSIFIER_TRANSFORM(crop_image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    confidence, pred_idx = probs.max(dim=1)
    return idx_to_class[pred_idx.item()], confidence.item()


def mask_to_crop(image_np, mask_entry):
    x, y, w, h = mask_entry["bbox_xywh"]
    crop_np = image_np[y : y + h, x : x + w]
    return Image.fromarray(crop_np)


def run_full_cv_pipeline(
    video_path,
    sam_checkpoint,
    sam_model_type,
    classifier_checkpoint,
    claim_id=None,
    fps=2.0,
    max_frames=8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    claim_id = claim_id or f"CLM-{uuid.uuid4().hex[:12]}"

    frame_paths = extract_frames(
        video_path, out_dir=f"/tmp/{claim_id}/frames", target_fps=fps
    )
    frame_paths = frame_paths[:max_frames]

    mask_generator = load_sam(sam_checkpoint, sam_model_type, device)
    classifier, idx_to_class = load_classifier(classifier_checkpoint, device)

    raw_detections = []
    for frame_index, frame_path in enumerate(frame_paths):
        image_pil = Image.open(frame_path).convert("RGB")
        image_np = np.array(image_pil)

        masks = mask_generator.generate(image_np)
        for mask in masks:
            crop = mask_to_crop(image_np, {"bbox_xywh": mask["bbox"]})
            if crop.size[0] < 10 or crop.size[1] < 10:
                continue

            damage_type, confidence = classify_crop(
                classifier, idx_to_class, crop, device
            )
            if damage_type == "no_damage":
                continue

            raw_detections.append(
                {
                    "frame_id": os.path.basename(frame_path),
                    "frame_index": frame_index,
                    "part": "unknown",
                    "damage_type": damage_type,
                    "confidence": confidence,
                    "bbox": mask["bbox"],
                    "estimated_surface_area_cm2": mask["area"] * 0.05,
                }
            )

    fused = run_fusion_pipeline(raw_detections)
    thresholded = apply_confidence_thresholds(fused)

    return {
        "claim_id": claim_id,
        "video_id": os.path.basename(video_path),
        "damage_findings": thresholded["damage_findings"],
        "low_confidence_review_needed": thresholded["low_confidence_review_needed"],
        "model_version": "cv-pipeline-week3-v0.1.0",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_model_type", default="vit_b")
    parser.add_argument("--classifier_checkpoint", required=True)
    parser.add_argument("--out_json", default="outputs/week3_day2/damage_report.json")
    args = parser.parse_args()

    result = run_full_cv_pipeline(
        args.video, args.sam_checkpoint, args.sam_model_type, args.classifier_checkpoint
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
