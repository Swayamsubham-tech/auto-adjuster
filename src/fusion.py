"""
Week 3, Day 1 — Cross-frame fusion.
"""

import argparse
import json
from collections import defaultdict


def iou_2d(box_a, box_b):
    """Standard IoU. box format: [x, y, w, h]. Used WITHIN a single frame to
    suppress duplicate/overlapping SAM masks of the same object."""
    ax1, ay1, aw, ah = box_a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = box_b
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def suppress_within_frame_duplicates(detections, iou_threshold=0.7):
    """Within a SINGLE frame: same (part, damage_type) + high IoU -> keep only
    the higher-confidence one. Handles SAM sometimes emitting near-duplicate masks."""
    by_frame = defaultdict(list)
    for d in detections:
        by_frame[d["frame_id"]].append(d)

    kept = []
    for frame_id, frame_dets in by_frame.items():
        frame_dets = sorted(frame_dets, key=lambda d: -d["confidence"])
        suppressed = [False] * len(frame_dets)

        for i in range(len(frame_dets)):
            if suppressed[i]:
                continue
            kept.append(frame_dets[i])
            for j in range(i + 1, len(frame_dets)):
                if suppressed[j]:
                    continue
                same_class = (
                    frame_dets[i]["part"] == frame_dets[j]["part"]
                    and frame_dets[i]["damage_type"] == frame_dets[j]["damage_type"]
                )
                is_overlap = (
                    iou_2d(frame_dets[i]["bbox"], frame_dets[j]["bbox"])
                    >= iou_threshold
                )
                if same_class and is_overlap:
                    suppressed[j] = True
    return kept


def fuse_across_frames(detections, frame_window=3):
    """
    Groups by (part, damage_type); merges detections within `frame_window`
    frames of each other into one fused finding. Fused confidence = MAX across
    members (trust the clearest view). Area = average across members.
    """
    detections = sorted(detections, key=lambda d: d["frame_index"])
    by_class = defaultdict(list)
    for d in detections:
        by_class[(d["part"], d["damage_type"])].append(d)

    fused_findings = []
    finding_counter = 0

    for (part, damage_type), group in by_class.items():
        group = sorted(group, key=lambda d: d["frame_index"])
        clusters = []
        current_cluster = [group[0]]

        for det in group[1:]:
            if det["frame_index"] - current_cluster[-1]["frame_index"] <= frame_window:
                current_cluster.append(det)
            else:
                clusters.append(current_cluster)
                current_cluster = [det]
        clusters.append(current_cluster)

        for cluster in clusters:
            finding_counter += 1
            confidences = [d["confidence"] for d in cluster]
            areas = [d.get("estimated_surface_area_cm2", 0) for d in cluster]

            fused_findings.append(
                {
                    "finding_id": f"f{finding_counter}",
                    "part": part,
                    "damage_type": damage_type,
                    "confidence": max(confidences),
                    "num_frames_observed": len(cluster),
                    "frames_observed": [d["frame_id"] for d in cluster],
                    "estimated_surface_area_cm2": (
                        sum(areas) / len(areas) if areas else None
                    ),
                }
            )

    return fused_findings


def run_fusion_pipeline(raw_detections, iou_threshold=0.7, frame_window=3):
    deduped = suppress_within_frame_duplicates(raw_detections, iou_threshold)
    return fuse_across_frames(deduped, frame_window)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections_json", required=True)
    parser.add_argument("--out_json", default="outputs/week3_day1/fused_findings.json")
    parser.add_argument("--iou_threshold", type=float, default=0.7)
    parser.add_argument("--frame_window", type=int, default=3)
    args = parser.parse_args()

    with open(args.detections_json) as f:
        raw_detections = json.load(f)

    fused = run_fusion_pipeline(raw_detections, args.iou_threshold, args.frame_window)

    import os

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(fused, f, indent=2)

    print(
        f"Fused {len(raw_detections)} raw detections into {len(fused)} unique findings."
    )
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
