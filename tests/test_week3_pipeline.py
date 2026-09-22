"""
Week 3, Day 6 — Integration tests, including a mocked full-pipeline test.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

from PIL import Image

# --- AGGRESSIVE MOCKING FOR CI ---
# Prevent CI runner from crashing when it encounters heavy ML imports in our source files.
_mock_modules = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils",
    "torch.utils.data",
    "torch.optim",
    "torchvision",
    "torchvision.transforms",
    "torchvision.models",
    "matplotlib",
    "matplotlib.pyplot",
    "cv2",
    "segment_anything",
]
for _mod in _mock_modules:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from confidence import apply_confidence_thresholds  # noqa: E402
from fusion import run_fusion_pipeline  # noqa: E402


def test_fusion_then_confidence_end_to_end():
    raw_detections = [
        {
            "frame_id": "f0",
            "frame_index": 0,
            "part": "front_bumper",
            "damage_type": "dent",
            "confidence": 0.9,
            "bbox": [10, 10, 50, 50],
            "estimated_surface_area_cm2": 300,
        },
        {
            "frame_id": "f1",
            "frame_index": 1,
            "part": "front_bumper",
            "damage_type": "dent",
            "confidence": 0.85,
            "bbox": [11, 10, 50, 50],
            "estimated_surface_area_cm2": 310,
        },
        {
            "frame_id": "f5",
            "frame_index": 5,
            "part": "left_headlight",
            "damage_type": "shattered_lens",
            "confidence": 0.3,
            "bbox": [200, 50, 40, 40],
            "estimated_surface_area_cm2": 140,
        },
    ]
    fused = run_fusion_pipeline(raw_detections)
    assert len(fused) == 2

    result = apply_confidence_thresholds(fused)
    assert result["low_confidence_review_needed"] is True
    for finding in result["damage_findings"]:
        assert finding["confidence_tier"] in {"high", "medium", "low"}


def test_full_damage_json_schema_shape():
    raw_detections = [
        {
            "frame_id": "f0",
            "frame_index": 0,
            "part": "front_bumper",
            "damage_type": "dent",
            "confidence": 0.9,
            "bbox": [10, 10, 50, 50],
            "estimated_surface_area_cm2": 300,
        }
    ]
    fused = run_fusion_pipeline(raw_detections)
    thresholded = apply_confidence_thresholds(fused)

    damage_json = {
        "claim_id": "CLM-TEST-001",
        "video_id": "test_video.mp4",
        "damage_findings": thresholded["damage_findings"],
        "low_confidence_review_needed": thresholded["low_confidence_review_needed"],
        "model_version": "cv-pipeline-week3-v0.1.0",
    }

    required_top = {
        "claim_id",
        "video_id",
        "damage_findings",
        "low_confidence_review_needed",
        "model_version",
    }
    assert required_top.issubset(damage_json.keys())

    required_finding_keys = {
        "finding_id",
        "part",
        "damage_type",
        "confidence",
        "confidence_tier",
    }
    for finding in damage_json["damage_findings"]:
        assert required_finding_keys.issubset(finding.keys())

    json_str = json.dumps(damage_json)
    assert json.loads(json_str)["claim_id"] == "CLM-TEST-001"


def test_mocked_full_pipeline_wiring():
    """Replaces SAM, the classifier, and frame extraction with fakes."""
    from pipeline import run_full_cv_pipeline

    with tempfile.TemporaryDirectory() as tmp:
        fake_frame_paths = []
        for i in range(3):
            path = os.path.join(tmp, f"frame_{i:04d}.jpg")
            Image.new("RGB", (100, 100), (i * 30, 50, 100)).save(path)
            fake_frame_paths.append(path)

        fake_mask_generator = MagicMock()
        fake_mask_generator.generate.return_value = [
            {
                "bbox": [10, 10, 40, 40],
                "area": 1600,
                "predicted_iou": 0.9,
                "stability_score": 0.9,
            },
        ]

        with patch("pipeline.extract_frames", return_value=fake_frame_paths), patch(
            "pipeline.load_sam", return_value=fake_mask_generator
        ), patch(
            "pipeline.load_classifier", return_value=(MagicMock(), {0: "dent"})
        ), patch(
            "pipeline.classify_crop", return_value=("dent", 0.88)
        ):
            result = run_full_cv_pipeline(
                video_path="fake_video.mp4",
                sam_checkpoint="fake.pth",
                sam_model_type="vit_b",
                classifier_checkpoint="fake_classifier.pth",
                claim_id="CLM-MOCKTEST",
                max_frames=3,
            )

    assert result["claim_id"] == "CLM-MOCKTEST"
    assert len(result["damage_findings"]) >= 1
    assert result["damage_findings"][0]["damage_type"] == "dent"
    assert result["damage_findings"][0]["num_frames_observed"] == 3
