import cv2
import json
import numpy as np
from pathlib import Path

# =========================
# Paths
# =========================
FACE_DIR = Path("processed/faces")
LANDMARK_DIR = Path("processed/landmarks")
MASK_DIR = Path("processed/masks")

MASK_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# MediaPipe Face Mesh
# Nose landmark indices
# =========================
NOSE_POINTS = [
    1, 2, 4, 5, 6, 19, 44, 45, 48, 49,
    64, 94, 97, 98, 115, 122, 128, 131,
    134, 168, 174, 188, 193, 197, 236, 237,
    240, 241, 242, 243, 244, 245, 274, 278,
    279, 281, 282, 283, 289, 290, 294, 305,
    306, 309, 326, 327, 344, 351, 363,
    370, 399, 420, 440
]

# =========================
# Process
# =========================
json_files = list(LANDMARK_DIR.glob("*.json"))

print(f"Found {len(json_files)} landmark files.")

success = 0
failed = 0

for json_path in json_files:

    image_id = json_path.stem

    image_path = FACE_DIR / f"{image_id}.png"

    if not image_path.exists():
        image_path = FACE_DIR / f"{image_id}.jpg"

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_id}")
        failed += 1
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Cannot read: {image_path}")
        failed += 1
        continue

    h, w = image.shape[:2]

    # Load landmarks
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    landmarks = data["landmarks"]

    points = []

    for idx in NOSE_POINTS:

        if idx >= len(landmarks):
            continue

        x = int(landmarks[idx]["x"] * w)
        y = int(landmarks[idx]["y"] * h)

        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        points.append([x, y])

    if len(points) < 3:
        print(f"[FAILED] Not enough nose points: {image_id}")
        failed += 1
        continue

    points = np.array(points, dtype=np.int32)

    # =========================
    # Create mask
    # =========================
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillConvexPoly(
        mask,
        points,
        255
    )

    # =========================
    # Expand mask slightly
    # =========================
    kernel = np.ones((15, 15), np.uint8)

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    # =========================
    # Save
    # =========================
    output_path = MASK_DIR / f"{image_id}.png"

    cv2.imwrite(
        str(output_path),
        mask
    )

    print(f"[OK] {image_id} -> {output_path}")

    success += 1

print("\n==============================")
print("NOSE MASK COMPLETE")
print("==============================")
print(f"Success : {success}")
print(f"Failed  : {failed}")
print(f"Total   : {success + failed}")