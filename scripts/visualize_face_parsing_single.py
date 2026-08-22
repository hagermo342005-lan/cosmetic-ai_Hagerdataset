from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# PATHS
# ============================================================

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

OUTPUT_PATH = (
    ROOT
    / "processed"
    / "test_face_parsing_0_colored.jpg"
)


# ============================================================
# COLORS FOR 19 CLASSES
# ============================================================

COLORS = np.array([
    [0,   0,   0],
    [255, 0,   0],
    [0,   255, 0],
    [0,   0,   255],
    [255, 255, 0],
    [255, 0,   255],
    [0,   255, 255],
    [128, 0,   0],
    [0,   128, 0],
    [0,   0,   128],
    [128, 128, 0],
    [128, 0,   128],
    [0,   128, 128],
    [255, 128, 0],
    [255, 0,   128],
    [128, 255, 0],
    [0,   255, 128],
    [128, 128, 255],
    [255, 128, 128],
], dtype=np.uint8)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        f"Could not read:\n{IMAGE_PATH}"
    )

original_height, original_width = image.shape[:2]


# ============================================================
# PREPROCESS
# ============================================================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

resized = cv2.resize(
    image_rgb,
    (512, 512)
)

input_image = (
    resized.astype(np.float32)
    / 255.0
)

input_image = np.transpose(
    input_image,
    (2, 0, 1)
)

input_image = input_image[
    None,
    ...
]


# ============================================================
# LOAD MODEL
# ============================================================

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


# ============================================================
# INFERENCE
# ============================================================

outputs = session.run(
    None,
    {
        input_name: input_image
    }
)

logits = outputs[0]

segmentation = np.argmax(
    logits[0],
    axis=0
).astype(np.uint8)


# ============================================================
# RESIZE MASK TO ORIGINAL IMAGE
# ============================================================

segmentation = cv2.resize(
    segmentation,
    (original_width, original_height),
    interpolation=cv2.INTER_NEAREST
)


# ============================================================
# CREATE COLOR MASK
# ============================================================

colored_mask = COLORS[
    segmentation
]


# ============================================================
# BLEND WITH ORIGINAL IMAGE
# ============================================================

overlay = cv2.addWeighted(
    image,
    0.55,
    colored_mask,
    0.45,
    0
)


# ============================================================
# DRAW CLASS NUMBERS
# ============================================================

classes = np.unique(
    segmentation
)

for class_id in classes:

    if class_id == 0:
        continue

    ys, xs = np.where(
        segmentation == class_id
    )

    if len(xs) == 0:
        continue

    center_x = int(
        np.mean(xs)
    )

    center_y = int(
        np.mean(ys)
    )

    cv2.putText(
        overlay,
        f"Class {class_id}",
        (center_x, center_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    str(OUTPUT_PATH),
    overlay
)


print("=" * 70)
print("FACE PARSING VISUALIZATION")
print("=" * 70)

print(
    f"Image    : {IMAGE_PATH}"
)

print(
    f"Output   : {OUTPUT_PATH}"
)

print(
    f"Classes  : {list(classes)}"
)

print("=" * 70)