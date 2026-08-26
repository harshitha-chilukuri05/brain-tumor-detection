"""
evaluate.py
-----------
Loads a trained checkpoint and produces a full evaluation report on the
test set: accuracy, per-class precision/recall/F1, and a confusion matrix
plot. This is the kind of honest, detailed evaluation that separates a
portfolio project from a notebook that just prints "accuracy: 0.97".

Usage:
    python src/evaluate.py --data_dir data --model_path saved_models/best_model.pt
"""

import argparse
import os

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from model import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate brain tumor MRI classifier")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_path", type=str, default="saved_models/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(args.model_path, device=device)

    _, test_loader, _ = get_dataloaders(args.data_dir, batch_size=args.batch_size)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Text report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Brain Tumor MRI Classification")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)

    print(f"\nClassification report saved to {args.output_dir}/classification_report.txt")
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
