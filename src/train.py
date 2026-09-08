"""
Week 2, Day 4 — Training loop with MLflow experiment tracking.
"""

import argparse
import os

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

sys.path.insert(0, os.path.dirname(__file__))
from dataset import DamageCropDataset  # noqa: E402
from model import build_damage_classifier, count_trainable_params  # noqa: E402
from sampler_utils import build_weighted_sampler  # noqa: E402


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--freeze_backbone", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument(
        "--use_weighted_sampler", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument("--out_dir", default="outputs/week2_training")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = DamageCropDataset(args.manifest, split="train")
    val_ds = DamageCropDataset(
        args.manifest, split="val", class_to_idx=train_ds.class_to_idx
    )
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"Classes: {train_ds.class_to_idx}")

    if args.use_weighted_sampler:
        sampler = build_weighted_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_damage_classifier(
        num_classes=len(train_ds.class_to_idx),
        freeze_backbone=args.freeze_backbone,
        pretrained=True,
    ).to(device)
    print(f"Trainable parameters: {count_trainable_params(model):,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    mlflow_db_path = os.path.join(os.path.abspath(args.out_dir), "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment("auto-adjuster-damage-classifier")

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(args.out_dir, "best_model.pth")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "freeze_backbone": args.freeze_backbone,
                "num_classes": len(train_ds.class_to_idx),
                "train_size": len(train_ds),
                "val_size": len(val_ds),
            }
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device
            )
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

            print(
                f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_to_idx": train_ds.class_to_idx,
                        "epoch": epoch,
                        "val_acc": val_acc,
                    },
                    best_checkpoint_path,
                )
                print(f"  -> New best model saved (val_acc={val_acc:.3f})")

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_artifact(best_checkpoint_path)

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.3f}")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
