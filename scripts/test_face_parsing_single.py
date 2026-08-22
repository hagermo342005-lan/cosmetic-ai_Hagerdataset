from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    ROOT
    / "processed"
    / "faces"
    / "CelebAMask-HQ"
    / "0.jpg"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "face_parsing"
    / "parsing_resnet18.onnx"
)


print("=" * 70)
print("SINGLE IMAGE FACE PARSING TEST")
print("=" * 70)

print(f"Image : {IMAGE_PATH}")
print(f"Model : {MODEL_PATH}")


# ------------------------------------------------------------
# Load image
# ------------------------------------------------------------

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        f"Could not read image:\n{IMAGE_PATH}"
    )


original_height, original_width = image.shape[:2]

print(
    f"Original size : "
    f"{original_width} x {original_height}"
)


# ------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image_resized = cv2.resize(
    image_rgb,
    (512, 512)
)

image_float = (
    image_resized
    .astype(np.float32)
    / 255.0
)

# HWC -> CHW
image_input = np.transpose(
    image_float,
    (2, 0, 1)
)

# Add batch
image_input = image_input[
    None,
    ...
]


print(
    f"Model input : "
    f"{image_input.shape}"
)


# ------------------------------------------------------------
# Load ONNX
# ------------------------------------------------------------

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print(
    f"Input name : {input_name}"
)


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

outputs = session.run(
    None,
    {
        input_name: image_input
    }
)


print()
print(
    f"Number of outputs : {len(outputs)}"
)


for i, output in enumerate(outputs):

    print(
        f"Output {i}: "
        f"shape={output.shape}, "
        f"dtype={output.dtype}, "
        f"min={output.min():.4f}, "
        f"max={output.max():.4f}"
    )


# ------------------------------------------------------------
# Use first output
# ------------------------------------------------------------

logits = outputs[0]

# Shape:
# (1, 19, 512, 512)

segmentation = np.argmax(
    logits[0],
    axis=0
).astype(np.uint8)


# ------------------------------------------------------------
# Show classes detected
# ------------------------------------------------------------

classes, counts = np.unique(
    segmentation,
    return_counts=True
)


print()
print("=" * 70)
print("DETECTED CLASS IDs")
print("=" * 70)

for class_id, count in zip(
    classes,
    counts
):

    percentage = (
        count
        /
        segmentation.size
        *
        100
    )

    print(
        f"Class {int(class_id):2d} : "
        f"{int(count):7d} pixels "
        f"({percentage:6.2f}%)"
    )


# ------------------------------------------------------------
# Save raw mask
# ------------------------------------------------------------

output_path = (
    ROOT
    / "processed"
    / "test_face_parsing_0_raw.png"
)

cv2.imwrite(
    str(output_path),
    segmentation
)


print()
print(
    f"Raw mask saved to:\n{output_path}"
)

print("=" * 70)