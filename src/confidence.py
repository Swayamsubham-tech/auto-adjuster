"""
Week 3, Day 3 — Confidence thresholding.
"""

import argparse
import json

DEFAULT_HIGH_THRESHOLD = 0.80
DEFAULT_LOW_THRESHOLD = 0.50


def classify_confidence(
    confidence,
    high_threshold=DEFAULT_HIGH_THRESHOLD,
    low_threshold=DEFAULT_LOW_THRESHOLD,
):
    if confidence >= high_threshold:
        return "high"
    elif confidence >= low_threshold:
        return "medium"
    else:
        return "low"


def apply_confidence_thresholds(
    fused_findings,
    high_threshold=DEFAULT_HIGH_THRESHOLD,
    low_threshold=DEFAULT_LOW_THRESHOLD,
):
    annotated = []
    any_low = False
    any_medium = False

    for finding in fused_findings:
        tier = classify_confidence(finding["confidence"], high_threshold, low_threshold)
        finding_copy = dict(finding)
        finding_copy["confidence_tier"] = tier
        annotated.append(finding_copy)

        if tier == "low":
            any_low = True
        elif tier == "medium":
            any_medium = True

    return {
        "damage_findings": annotated,
        "low_confidence_review_needed": any_low,
        "medium_confidence_present": any_medium,
        "thresholds_used": {"high": high_threshold, "low": low_threshold},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused_json", required=True)
    parser.add_argument(
        "--out_json", default="outputs/week3_day3/thresholded_findings.json"
    )
    parser.add_argument("--high_threshold", type=float, default=DEFAULT_HIGH_THRESHOLD)
    parser.add_argument("--low_threshold", type=float, default=DEFAULT_LOW_THRESHOLD)
    args = parser.parse_args()

    with open(args.fused_json) as f:
        fused_findings = json.load(f)

    result = apply_confidence_thresholds(
        fused_findings, args.high_threshold, args.low_threshold
    )

    import os

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Findings: {len(result['damage_findings'])}")
    print(f"low_confidence_review_needed: {result['low_confidence_review_needed']}")


if __name__ == "__main__":
    main()
