import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "resnet18_beauty_best.pth"


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Predict beauty score for a single face image"
)

parser.add_argument(
    "--image",
    required=True,
    help="Path to input image"
)

args = parser.parse_args()


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Single Image Beauty Prediction")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Image
# ============================================================

image_path = Path(args.image)

if not image_path.is_absolute():
    image_path = ROOT / image_path

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found:\n{image_path}"
    )

print(f"\nImage:")
print(image_path)


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Load model
# ============================================================

print("\nLoading ResNet18...")

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# Predict
# ============================================================

print("\nRunning prediction...")

image = Image.open(image_path).convert("RGB")

input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)

with torch.no_grad():

    output = model(input_tensor)

    score = output.item()


# ============================================================
# Clamp score
# ============================================================

score = max(1.0, min(5.0, score))


# ============================================================
# Result
# ============================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(f"Beauty Score: {score:.2f} / 5.00")

print("=" * 60)