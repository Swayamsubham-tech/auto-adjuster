"""
Week 2, Day 7 — Evaluation: confusion matrix, per-class metrics, error analysis.
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import sys

sys.path.insert(0, os.path.dirname(__file__))
from dataset import DamageCropDataset
from model import build_damage_classifier


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_confidences = [], [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, preds = probs.max(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_confidences.extend(confidences.cpu().tolist())
    return all_preds, all_labels, all_confidences


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(
        figsize=(max(6, len(class_names)), max(5, len(class_names) * 0.8))
    )
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", default="outputs/week2_eval")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_acc={checkpoint['val_acc']:.3f})"
    )

    ds = DamageCropDataset(args.manifest, split=args.split, class_to_idx=class_to_idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = build_damage_classifier(
        num_classes=len(class_to_idx), freeze_backbone=True, pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    preds, labels, confidences = get_predictions(model, loader, device)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )
    overall_acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)

    print(f"\n=== Overall accuracy: {overall_acc:.3f} ===\n")
    print(
        f"{'Class':20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}"
    )
    report_rows = []
    for i, cname in enumerate(class_names):
        print(
            f"{cname:20s} {precision[i]:10.3f} {recall[i]:10.3f} {f1[i]:10.3f} {support[i]:10d}"
        )
        report_rows.append(
            {
                "class": cname,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        )

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"\nSaved confusion matrix: {cm_path}")

    errors = [
        {
            "index": i,
            "true": idx_to_class[labels[i]],
            "predicted": idx_to_class[preds[i]],
            "confidence": confidences[i],
        }
        for i in range(len(labels))
        if preds[i] != labels[i]
    ]
    errors.sort(key=lambda e: -e["confidence"])

    print(f"\n=== {len(errors)} misclassifications out of {len(labels)} ===")
    print("Top 10 highest-confidence mistakes (most concerning, review these first):")
    for e in errors[:10]:
        print(
            f"  true={e['true']:15s} predicted={e['predicted']:15s} confidence={e['confidence']:.3f}"
        )

    report_path = os.path.join(args.out_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "split": args.split,
                "overall_accuracy": overall_acc,
                "per_class": report_rows,
                "confusion_matrix": cm.tolist(),
                "class_names": class_names,
                "top_errors": errors[:20],
            },
            f,
            indent=2,
        )
    print(f"\nFull report saved: {report_path}")


if __name__ == "__main__":
    main()
