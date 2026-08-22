import cv2
import json
import numpy as np
from pathlib import Path

# =========================================================
# INPUT
# =========================================================

IMAGE_PATH = Path("processed/faces/00000.png")
LANDMARK_PATH = Path("processed/landmarks/00000.json")

OUT_DIR = Path("processed/nose_precise_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD
# =========================================================

img = cv2.imread(str(IMAGE_PATH))

if img is None:
    raise RuntimeError("Image not found")

h, w = img.shape[:2]

with open(LANDMARK_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

landmarks = data["landmarks"]

print("Image:", IMAGE_PATH)
print("Size:", w, "x", h)
print("Landmarks:", len(landmarks))


# =========================================================
# MEDIAPIPE NOSE LANDMARKS
#
# These points describe the nose:
# bridge + sides + tip + nostril region
# =========================================================

NOSE_IDS = [
    1,      # nose center/tip
    2,
    4,
    5,
    6,
    19,
    20,
    44,
    45,
    48,
    49,
    51,
    94,
    97,
    98,
    115,
    122,
    129,
    168,
    195,
    197,
    236,
    240,
    278,
    290,
    326,
    327,
    344,
    351,
    419,
    420,
    429,
    437,
    440,
    456
]


# =========================================================
# CONVERT NORMALIZED -> PIXELS
# =========================================================

nose_points = []

for idx in NOSE_IDS:

    p = landmarks[idx]

    x = int(p["x"] * w)
    y = int(p["y"] * h)

    if 0 <= x < w and 0 <= y < h:
        nose_points.append((x, y))


nose_points = np.array(
    nose_points,
    dtype=np.int32
)

print("Nose points:", len(nose_points))


# =========================================================
# IMAGE WITH NOSE LANDMARKS
# =========================================================

landmark_img = img.copy()

for i, (x, y) in enumerate(nose_points):

    cv2.circle(
        landmark_img,
        (x, y),
        3,
        (0, 255, 0),
        -1
    )

    cv2.putText(
        landmark_img,
        str(i),
        (x + 3, y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (0, 255, 0),
        1
    )


cv2.imwrite(
    str(OUT_DIR / "01_nose_landmarks.jpg"),
    landmark_img
)


# =========================================================
# CREATE NOSE MASK
# =========================================================

mask = np.zeros(
    (h, w),
    dtype=np.uint8
)


# Convex hull around nose landmarks
hull = cv2.convexHull(nose_points)

cv2.fillConvexPoly(
    mask,
    hull,
    255
)


# =========================================================
# IMPORTANT:
# Shrink mask slightly so it does NOT spill
# into cheeks / eyes / lips
# =========================================================

kernel = np.ones(
    (5, 5),
    np.uint8
)

mask = cv2.erode(
    mask,
    kernel,
    iterations=1
)


# =========================================================
# SAVE MASK
# =========================================================

cv2.imwrite(
    str(OUT_DIR / "02_nose_mask.png"),
    mask
)


# =========================================================
# MASK OUTLINE
# =========================================================

outline = img.copy()

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(
    outline,
    contours,
    -1,
    (0, 255, 0),
    2
)

cv2.imwrite(
    str(OUT_DIR / "03_nose_outline.jpg"),
    outline
)


# =========================================================
# OVERLAY
# =========================================================

overlay = img.copy()

red = np.zeros_like(img)
red[:, :, 2] = 255

m = mask > 0

overlay[m] = cv2.addWeighted(
    img[m],
    0.35,
    red[m],
    0.65,
    0
)

cv2.drawContours(
    overlay,
    contours,
    -1,
    (0, 255, 0),
    2
)

cv2.imwrite(
    str(OUT_DIR / "04_nose_mask_preview.jpg"),
    overlay
)


# =========================================================
# CROP NOSE REGION
# =========================================================

ys, xs = np.where(mask > 0)

if len(xs) > 0:

    x1 = max(0, xs.min() - 20)
    x2 = min(w, xs.max() + 20)

    y1 = max(0, ys.min() - 20)
    y2 = min(h, ys.max() + 20)

    nose_crop = img[
        y1:y2,
        x1:x2
    ]

    cv2.imwrite(
        str(OUT_DIR / "05_nose_crop.jpg"),
        nose_crop
    )


# =========================================================
# DONE
# =========================================================

print()
print("==============================")
print("PRECISE NOSE TEST COMPLETE")
print("==============================")
print("Output:")
print(OUT_DIR)
print()
print("Files:")
print("01_nose_landmarks.jpg")
print("02_nose_mask.png")
print("03_nose_outline.jpg")
print("04_nose_mask_preview.jpg")
print("05_nose_crop.jpg")
print("==============================")