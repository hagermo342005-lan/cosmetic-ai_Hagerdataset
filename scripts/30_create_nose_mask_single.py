import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "input" / "0003.png"

LANDMARKS_PATH = (
    BASE_DIR
    / "processed"
    / "landmarks_single"
    / "0003.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "single_test_0003"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

height, width = image.shape[:2]


print("=" * 70)
print("NOSE MASK - 0003")
print("=" * 70)

print(f"Image size: {width} x {height}")


# ============================================================
# LOAD LANDMARKS
# ============================================================

if not LANDMARKS_PATH.exists():
    raise FileNotFoundError(
        f"Landmarks not found:\n{LANDMARKS_PATH}"
    )


with open(
    LANDMARKS_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


landmarks = data["landmarks"]

print(
    f"Landmarks: {len(landmarks)}"
)


if len(landmarks) != 478:
    raise RuntimeError(
        f"Expected 478 landmarks, "
        f"got {len(landmarks)}"
    )


# ============================================================
# NOSE OUTER CONTOUR
# ============================================================

nose_contour_indices = [
    98,
    97,
    2,
    326,
    327,
    294,
    278,
    344,
    440,
    275,
    4,
    45,
    220,
    115,
    48,
    64
]


nose_contour = np.array(
    [
        (
            int(round(landmarks[i]["x"])),
            int(round(landmarks[i]["y"]))
        )
        for i in nose_contour_indices
    ],
    dtype=np.int32
)


# ============================================================
# CREATE MASK
# ============================================================

mask = np.zeros(
    (height, width),
    dtype=np.uint8
)


cv2.fillPoly(
    mask,
    [nose_contour],
    255
)


# Small expansion
kernel = np.ones(
    (5, 5),
    dtype=np.uint8
)

mask = cv2.dilate(
    mask,
    kernel,
    iterations=1
)


# Slight blur for smoother inpainting
mask = cv2.GaussianBlur(
    mask,
    (5, 5),
    0
)


# ============================================================
# SAVE BEFORE
# ============================================================

before_path = (
    OUTPUT_DIR
    / "before.jpg"
)

cv2.imwrite(
    str(before_path),
    image
)


# ============================================================
# SAVE MASK
# ============================================================

mask_path = (
    OUTPUT_DIR
    / "mask.png"
)

cv2.imwrite(
    str(mask_path),
    mask
)


# ============================================================
# MASK OVERLAY
# ============================================================

overlay = image.copy()

red_layer = np.zeros_like(image)

red_layer[:, :, 2] = 255

alpha = (
    mask.astype(np.float32)
    / 255.0
)


for c in range(3):

    overlay[:, :, c] = (
        image[:, :, c]
        * (1 - alpha * 0.45)
        +
        red_layer[:, :, c]
        * (alpha * 0.45)
    ).astype(np.uint8)


overlay_path = (
    OUTPUT_DIR
    / "mask_overlay.jpg"
)

cv2.imwrite(
    str(overlay_path),
    overlay
)


# ============================================================
# LANDMARK VISUALIZATION
# ============================================================

landmark_image = image.copy()


for index in nose_contour_indices:

    x = int(round(
        landmarks[index]["x"]
    ))

    y = int(round(
        landmarks[index]["y"]
    ))


    cv2.circle(
        landmark_image,
        (x, y),
        6,
        (0, 255, 0),
        -1
    )


    cv2.putText(
        landmark_image,
        str(index),
        (x + 6, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )


cv2.polylines(
    landmark_image,
    [nose_contour],
    True,
    (255, 0, 0),
    2
)


landmark_path = (
    OUTPUT_DIR
    / "nose_landmarks.jpg"
)

cv2.imwrite(
    str(landmark_path),
    landmark_image
)


# ============================================================
# SAVE INDICES
# ============================================================

indices_path = (
    OUTPUT_DIR
    / "nose_indices.json"
)


with open(
    indices_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "image": str(
                IMAGE_PATH
            ),
            "indices": (
                nose_contour_indices
            ),
            "image_size": [
                width,
                height
            ]
        },
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"Before         : {before_path}"
)

print(
    f"Mask           : {mask_path}"
)

print(
    f"Mask overlay   : {overlay_path}"
)

print(
    f"Nose landmarks : {landmark_path}"
)

print(
    f"Indices         : {indices_path}"
)

print("=" * 70)