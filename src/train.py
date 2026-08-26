"""
train.py
--------
Trains the brain tumor classifier and saves the best checkpoint
(by validation/test accuracy) to saved_models/.

Usage:
    python src/train.py --data_dir data --epochs 15 --batch_size 32 --lr 1e-4

Produces:
    saved_models/best_model.pt         - best checkpoint (weights + class names)
    outputs/training_curves.png        - loss/accuracy curves
    outputs/training_log.csv           - per-epoch metrics
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train brain tumor MRI classifier")
    parser.add_argument("--data_dir", type=str, default="data", help="Path with train/ and test/ subfolders")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true", help="Only train the classifier head")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_dir", type=str, default="saved_models")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    model = build_model(num_classes=len(class_names), freeze_backbone=args.freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

        scheduler.step(test_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, class_names, os.path.join(args.model_dir, "best_model.pt"))
            print(f"  -> New best model saved (test_acc={best_acc:.4f})")

    # Save training log as CSV
    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        for i in range(args.epochs):
            writer.writerow([i + 1, history["train_loss"][i], history["train_acc"][i],
                              history["test_loss"][i], history["test_acc"][i]])

    # Plot curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["test_loss"], label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["test_acc"], label="Test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150)
    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Training curves saved to {args.output_dir}/training_curves.png")
    print(f"Best model saved to {args.model_dir}/best_model.pt")


if __name__ == "__main__":
    main()
