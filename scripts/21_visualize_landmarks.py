import json
import random
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FACES_DIR = BASE_DIR / "processed" / "faces"
LANDMARKS_DIR = BASE_DIR / "processed" / "landmarks"
OUTPUT_DIR = BASE_DIR / "processed" / "landmarks_preview"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

IMAGE_SIZE = 512
MAX_IMAGES = 5000

# نقطة كل كام Landmark نرسمها
DRAW_EVERY = 1


# ============================================================
# Helpers
# ============================================================

def load_landmarks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # نحاول دعم أكثر من شكل محتمل للـJSON
    if isinstance(data, dict):
        for key in ["landmarks", "points", "face_landmarks"]:
            if key in data:
                data = data[key]
                break

    return data


def extract_points(data):
    """
    يحول البيانات إلى قائمة [(x, y), ...]
    """

    points = []

    if not isinstance(data, list):
        return points

    for item in data:

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                x = float(item[0])
                y = float(item[1])
                points.append((x, y))
            except:
                pass

        elif isinstance(item, dict):
            if "x" in item and "y" in item:
                try:
                    points.append(
                        (float(item["x"]), float(item["y"]))
                    )
                except:
                    pass

    return points


def find_face_image(face_id):
    extensions = [".jpg", ".jpeg", ".png"]

    for ext in extensions:
        path = FACES_DIR / f"{face_id}{ext}"

        if path.exists():
            return path

    return None


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("FFHQ LANDMARK VISUALIZATION")
print("=" * 70)

print(f"Faces     : {FACES_DIR}")
print(f"Landmarks : {LANDMARKS_DIR}")
print(f"Output    : {OUTPUT_DIR}")
print()

json_files = sorted(LANDMARKS_DIR.glob("*.json"))

print(f"Landmark JSON files found: {len(json_files)}")
print()


success = 0
failed = 0
missing_images = 0
invalid_landmarks = 0


for index, json_path in enumerate(json_files[:MAX_IMAGES], start=1):

    face_id = json_path.stem

    try:

        # ----------------------------------------------------
        # Find corresponding face
        # ----------------------------------------------------

        image_path = find_face_image(face_id)

        if image_path is None:
            missing_images += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            failed += 1
            continue

        # ----------------------------------------------------
        # Load landmarks
        # ----------------------------------------------------

        data = load_landmarks(json_path)
        points = extract_points(data)

        if len(points) == 0:
            invalid_landmarks += 1
            continue

        # ----------------------------------------------------
        # Draw landmarks
        # ----------------------------------------------------

        preview = image.copy()

        h, w = preview.shape[:2]

        for i, (x, y) in enumerate(points):

            if i % DRAW_EVERY != 0:
                continue

            # لو الـcoordinates normalized من 0 إلى 1
            if 0 <= x <= 1 and 0 <= y <= 1:
                px = int(x * w)
                py = int(y * h)

            else:
                px = int(round(x))
                py = int(round(y))

            if 0 <= px < w and 0 <= py < h:

                cv2.circle(
                    preview,
                    (px, py),
                    2,
                    (0, 255, 0),
                    -1
                )

        # ----------------------------------------------------
        # Add information
        # ----------------------------------------------------

        text = f"ID: {face_id} | Landmarks: {len(points)}"

        cv2.putText(
            preview,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = OUTPUT_DIR / f"{face_id}_landmarks.jpg"

        cv2.imwrite(
            str(output_path),
            preview,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        success += 1

        if index % 100 == 0:
            print(
                f"Processed {index:,}/{min(len(json_files), MAX_IMAGES):,} "
                f"| Success: {success:,} "
                f"| Failed: {failed:,}"
            )

    except Exception as e:

        failed += 1

        print(
            f"Failed: {face_id} -> {type(e).__name__}: {e}"
        )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("LANDMARK VISUALIZATION COMPLETED")
print("=" * 70)

print(f"JSON files found      : {len(json_files)}")
print(f"Processed             : {min(len(json_files), MAX_IMAGES)}")
print(f"Visualization success : {success}")
print(f"Failed                : {failed}")
print(f"Missing images        : {missing_images}")
print(f"Invalid landmarks     : {invalid_landmarks}")

print()
print("Output:")
print(OUTPUT_DIR)

print("=" * 70)