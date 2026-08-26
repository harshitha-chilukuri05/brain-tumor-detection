"""
model.py
--------
Defines the classifier: a ResNet-18 backbone pretrained on ImageNet,
fine-tuned for brain tumor MRI classification.

Using transfer learning is a deliberate, defensible choice here (and worth
mentioning in interviews): MRI datasets are small relative to what's needed
to train a CNN from scratch, so we leverage low-level features (edges,
textures, gradients) already learned on ImageNet and re-train the deeper
layers + final classifier head on our domain-specific data.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """Build a ResNet-18 model adapted for num_classes outputs.

    Args:
        num_classes: number of output classes (e.g. 4: glioma, meningioma,
            notumor, pituitary).
        freeze_backbone: if True, freeze all convolutional layers and only
            train the final classifier head (faster, less prone to
            overfitting on small datasets, but lower ceiling on accuracy).

    Returns:
        A torch.nn.Module ready for training.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer to match our number of classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def save_checkpoint(model, class_names, path):
    """Save model weights + class name mapping together so predict.py
    doesn't need to guess the class order later."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, path)


def load_checkpoint(path, device="cpu"):
    """Load a model checkpoint saved with save_checkpoint()."""
    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
