import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

IMAGE_PATH = Path("input/00002.png")
LANDMARKS_PATH = Path("processed/landmarks_ffhq/00002.json")
OUTPUT_DIR = Path("output/single_test")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

height, width = image.shape[:2]

print("=" * 65)
print("NOSE LANDMARK + MASK TEST")
print("=" * 65)
print(f"Image size: {width} x {height}")

# ============================================================
# LOAD LANDMARK JSON
# ============================================================

with open(
    LANDMARKS_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

landmarks = data["landmarks"]

source_width = data["image_width"]
source_height = data["image_height"]

print(f"Landmark size: {source_width} x {source_height}")
print(f"Landmarks: {len(landmarks)}")

# ============================================================
# SCALE LANDMARKS
# ============================================================

scale_x = width / source_width
scale_y = height / source_height

points = []

for p in landmarks:

    x = int(round(p["x"] * scale_x))
    y = int(round(p["y"] * scale_y))

    points.append((x, y))

# ============================================================
# NOSE OUTER CONTOUR
# ============================================================
#
# Ordered MediaPipe nose contour points.
# IMPORTANT:
# We do NOT use convexHull here.
#

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
        points[i]
        for i in nose_contour_indices
    ],
    dtype=np.int32
)

# ============================================================
# CREATE NOSE MASK
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

# Small expansion only
# We don't want to affect the cheeks/eyes.

kernel = np.ones(
    (5, 5),
    dtype=np.uint8
)

mask = cv2.dilate(
    mask,
    kernel,
    iterations=1
)

# Slight blur for future inpainting
mask = cv2.GaussianBlur(
    mask,
    (5, 5),
    0
)

# ============================================================
# SAVE BEFORE
# ============================================================

cv2.imwrite(
    str(OUTPUT_DIR / "before.jpg"),
    image
)

# ============================================================
# SAVE MASK
# ============================================================

cv2.imwrite(
    str(OUTPUT_DIR / "mask.png"),
    mask
)

# ============================================================
# CREATE MASK OVERLAY
# ============================================================

overlay = image.copy()

red_layer = np.zeros_like(image)
red_layer[:, :, 2] = 255

# Use mask as transparency
alpha = mask.astype(np.float32) / 255.0

for c in range(3):

    overlay[:, :, c] = (
        image[:, :, c] * (1 - alpha * 0.45)
        + red_layer[:, :, c] * (alpha * 0.45)
    ).astype(np.uint8)

cv2.imwrite(
    str(OUTPUT_DIR / "mask_overlay.jpg"),
    overlay
)

# ============================================================
# VISUALIZE NOSE LANDMARKS
# ============================================================

landmark_image = image.copy()

for index in nose_contour_indices:

    x, y = points[index]

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

# Draw contour

cv2.polylines(
    landmark_image,
    [nose_contour],
    True,
    (255, 0, 0),
    2
)

cv2.imwrite(
    str(OUTPUT_DIR / "nose_landmarks.jpg"),
    landmark_image
)

# ============================================================
# SAVE ALL NOSE INDICES
# ============================================================

with open(
    OUTPUT_DIR / "nose_indices.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "image": str(IMAGE_PATH),
            "indices": nose_contour_indices,
            "source_size": [
                source_width,
                source_height
            ],
            "output_size": [
                width,
                height
            ],
            "scale": [
                scale_x,
                scale_y
            ]
        },
        f,
        indent=2
    )

# ============================================================
# DONE
# ============================================================

print("-" * 65)
print("SUCCESS")
print("-" * 65)

print(
    f"Before:          {OUTPUT_DIR / 'before.jpg'}"
)

print(
    f"Mask:            {OUTPUT_DIR / 'mask.png'}"
)

print(
    f"Mask overlay:    {OUTPUT_DIR / 'mask_overlay.jpg'}"
)

print(
    f"Nose landmarks:  {OUTPUT_DIR / 'nose_landmarks.jpg'}"
)

print(
    f"Nose indices:    {OUTPUT_DIR / 'nose_indices.json'}"
)

print("=" * 65)