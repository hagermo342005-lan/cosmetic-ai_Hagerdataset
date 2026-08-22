import argparse
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "resnet18_beauty_best.pth"

OUTPUT_DIR = ROOT / "processed" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Detect face and predict beauty score"
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
print("Face Detection + Beauty Prediction")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Input image
# ============================================================

image_path = Path(args.image)

if not image_path.is_absolute():
    image_path = ROOT / image_path

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found:\n{image_path}"
    )

print(f"\nInput image:")
print(image_path)


# ============================================================
# Load image with OpenCV
# ============================================================

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(
        f"Could not read image:\n{image_path}"
    )

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# Face detector
# ============================================================

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(
    cascade_path
)

if face_detector.empty():
    raise RuntimeError(
        "Could not load OpenCV Haar Cascade."
    )


faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(80, 80)
)


print(f"\nFaces detected: {len(faces)}")


if len(faces) == 0:
    raise RuntimeError(
        "No face detected in the image."
    )


# ============================================================
# Select largest face
# ============================================================

faces = sorted(
    faces,
    key=lambda box: box[2] * box[3],
    reverse=True
)

x, y, w, h = faces[0]

print(
    f"Largest face: x={x}, y={y}, "
    f"width={w}, height={h}"
)


# ============================================================
# Add padding around face
# ============================================================

padding = 0.20

pad_x = int(w * padding)
pad_y = int(h * padding)

x1 = max(0, x - pad_x)
y1 = max(0, y - pad_y)

x2 = min(image.shape[1], x + w + pad_x)
y2 = min(image.shape[0], y + h + pad_y)


face_crop = image[y1:y2, x1:x2]


# ============================================================
# Save face crop
# ============================================================

crop_path = OUTPUT_DIR / "detected_face.jpg"

cv2.imwrite(
    str(crop_path),
    face_crop
)

print(f"\nFace crop saved to:")
print(crop_path)


# ============================================================
# Convert BGR -> RGB
# ============================================================

face_rgb = cv2.cvtColor(
    face_crop,
    cv2.COLOR_BGR2RGB
)

pil_image = Image.fromarray(
    face_rgb
)


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


input_tensor = transform(
    pil_image
)

input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)


# ============================================================
# Load ResNet18
# ============================================================

print("\nLoading ResNet18...")

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(
        checkpoint
    )

model = model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# Prediction
# ============================================================

print("\nRunning prediction...")

with torch.no_grad():

    output = model(
        input_tensor
    )

    score = output.item()


# Keep score inside SCUT range
score = max(
    1.0,
    min(5.0, score)
)


# ============================================================
# Save visualization with bounding box
# ============================================================

visualization = image.copy()

cv2.rectangle(
    visualization,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3
)

label = f"Beauty Score: {score:.2f}/5"

cv2.putText(
    visualization,
    label,
    (x1, max(40, y1 - 10)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 0),
    2,
    cv2.LINE_AA
)

visualization_path = (
    OUTPUT_DIR /
    "face_detection_prediction.jpg"
)

cv2.imwrite(
    str(visualization_path),
    visualization
)


# ============================================================
# Result
# ============================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(f"Faces detected : {len(faces)}")
print(f"Beauty Score   : {score:.2f} / 5.00")

print("\nFiles:")

print(f"Face crop:")
print(crop_path)

print(f"\nDetection result:")
print(visualization_path)

print("=" * 60)