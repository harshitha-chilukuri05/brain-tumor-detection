"""
dataset.py
----------
Handles loading the Brain Tumor MRI dataset using torchvision's ImageFolder,
with appropriate transforms for training and evaluation.

Expected directory structure (created automatically if you follow the
Kaggle download instructions in the README):

    data/
        train/
            glioma/
            meningioma/
            notumor/
            pituitary/
        test/
            glioma/
            meningioma/
            notumor/
            pituitary/
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 224  # standard input size for ResNet-family models

# ImageNet normalization stats (used because we fine-tune a pretrained ResNet)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True):
    """Return the appropriate torchvision transform pipeline.

    Training transforms include light augmentation (flips, rotation, slight
    color jitter) since MRI scans are grayscale-ish and we don't want to
    distort diagnostically relevant structure too aggressively.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    """Build train/test DataLoaders from an ImageFolder-style directory.

    Args:
        data_dir: path containing 'train' and 'test' subfolders.
        batch_size: batch size for both loaders.
        num_workers: dataloader worker processes.

    Returns:
        train_loader, test_loader, class_names (list[str])
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Expected '{train_dir}' and '{test_dir}' to exist. "
            "See README.md for dataset download instructions."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))

    class_names = train_dataset.classes  # alphabetical order

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, class_names
