"""
predict.py
----------
Single-image inference with Grad-CAM visualization.

Grad-CAM highlights *which pixels* the model used to make its decision,
overlaid as a heatmap on the original MRI. This matters a lot for medical
imaging portfolio pieces: it demonstrates you understand that a raw
accuracy number isn't enough for a high-stakes domain, and that model
interpretability is something you actively considered.

Usage:
    python src/predict.py --image path/to/scan.jpg --model_path saved_models/best_model.pt
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import get_transforms, NORM_MEAN, NORM_STD, IMAGE_SIZE
from model import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Predict tumor class for a single MRI image")
    parser.add_argument("--image", type=str, required=True, help="Path to an MRI image")
    parser.add_argument("--model_path", type=str, default="saved_models/best_model.pt")
    parser.add_argument("--output", type=str, default="outputs/gradcam_result.png",
                         help="Where to save the Grad-CAM overlay")
    return parser.parse_args()


def predict_with_gradcam(image_path: str, model, class_names, device="cpu"):
    """Run inference on a single image and generate a Grad-CAM overlay.

    Returns:
        predicted_class (str), confidence (float), overlay_image (np.ndarray, RGB uint8)
    """
    transform = get_transforms(train=False)
    raw_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # Forward pass for prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
    predicted_class = class_names[pred_idx]

    # Grad-CAM on the last conv block of ResNet-18
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (H, W)

    # Prepare the un-normalized image (0-1 float) for overlay
    resized = raw_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.array(resized).astype(np.float32) / 255.0

    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return predicted_class, confidence, overlay


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(args.model_path, device=device)

    pred_class, confidence, overlay = predict_with_gradcam(
        args.image, model, class_names, device=device
    )

    print(f"Prediction: {pred_class} (confidence: {confidence:.2%})")

    Image.fromarray(overlay).save(args.output)
    print(f"Grad-CAM overlay saved to {args.output}")


if __name__ == "__main__":
    main()
