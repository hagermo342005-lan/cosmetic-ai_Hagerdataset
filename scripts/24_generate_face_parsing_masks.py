```python
import os
import json
import cv2
import numpy as np

from uniface.model_store import set_cache_dir
from uniface.parsing import BiSeNet


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACES_DIR = os.path.join(BASE_DIR, "processed", "faces")
LANDMARKS_DIR = os.path.join(BASE_DIR, "processed", "landmarks_ffhq")

MASKS_DIR = os.path.join(BASE_DIR, "processed", "masks")
PREVIEW_DIR = os.path.join(BASE_DIR, "processed", "masks_preview")

MODEL_DIR = os.path.join(BASE_DIR, "models", "face_parsing")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)


# ============================================================
# USE LOCAL BISEnET MODEL
# ============================================================

set_cache_dir(MODEL_DIR)

parser = BiSeNet()


# ============================================================
# 19 FACE PARSING CLASSES
# ============================================================

CLASS_NAMES = {
    0: "background",
    1: "skin",
    2: "left_eyebrow",
    3: "right_eyebrow",
    4: "left_eye",
    5: "right_eye",
    6: "eyeglasses",
    7: "left_ear",
    8: "right_ear",
    9: "earring",
    10: "nose",
    11: "mouth",
    12: "upper_lip",
    13: "lower_lip",
    14: "neck",
    15: "necklace",
    16: "cloth",
    17: "hair",
    18: "hat",
}


# ============================================================
# COLORS FOR VISUALIZATION
# OpenCV uses BGR
# ============================================================

COLORS = {
    0: (0, 0, 0),          # background - black
    1: (180, 180, 180),    # skin - gray
    2: (255, 120, 0),      # left eyebrow
    3: (255, 170, 0),      # right eyebrow
    4: (255, 0, 255),      # left eye
    5: (180, 0, 255),      # right eye
    6: (0, 255, 255),      # eyeglasses
    7: (255, 180, 100),    # left ear
    8: (255, 100, 100),    # right ear
    9: (100, 255, 255),    # earring
    10: (0, 255, 0),       # NOSE - GREEN
    11: (0, 140, 255),     # mouth
    12: (0, 0, 255),       # UPPER LIP - RED
    13: (0, 0, 180),       # LOWER LIP - DARK RED
    14: (120, 120, 120),   # neck
    15: (0, 215, 255),     # necklace
    16: (100, 150, 200),   # cloth
    17: (80, 80, 80),      # hair
    18: (50, 50, 50),      # hat
}


# ============================================================
# VISUALIZE MASK
# ============================================================

def colorize_mask(mask):
    """
    Convert class-ID mask into a colored BGR image.
    """

    h, w = mask.shape

    output = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in COLORS.items():
        output[mask == class_id] = color

    return output


# ============================================================
# ADD LABELS
# ============================================================

def add_title(image, title):
    result = image.copy()

    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1], 35),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        result,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return result


# ============================================================
# CREATE LEGEND
# ============================================================

def create_legend(width=512):
    height = 420

    legend = np.zeros((height, width, 3), dtype=np.uint8)

    y = 25

    for class_id in range(19):

        color = COLORS[class_id]
        name = CLASS_NAMES[class_id]

        cv2.rectangle(
            legend,
            (10, y - 16),
            (35, y + 5),
            color,
            -1
        )

        cv2.putText(
            legend,
            f"{class_id}: {name}",
            (45, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y += 21

    return legend


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("FFHQ FACE PARSING / MASK GENERATION")
print("=" * 70)

print(f"Faces     : {FACES_DIR}")
print(f"Landmarks : {LANDMARKS_DIR}")
print(f"Masks     : {MASKS_DIR}")
print(f"Preview   : {PREVIEW_DIR}")
print("=" * 70)


# ------------------------------------------------------------
# Get valid landmark IDs
# ------------------------------------------------------------

landmark_files = [
    f for f in os.listdir(LANDMARKS_DIR)
    if f.lower().endswith(".json")
]

valid_ids = set()

for filename in landmark_files:

    image_id = os.path.splitext(filename)[0]

    json_path = os.path.join(
        LANDMARKS_DIR,
        filename
    )

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        landmarks = data.get("landmarks", [])

        if len(landmarks) > 0:
            valid_ids.add(image_id)

    except Exception:
        pass


# ------------------------------------------------------------
# Get FFHQ images
# ------------------------------------------------------------

image_files = [
    f for f in os.listdir(FACES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_files.sort()

print(f"Images found       : {len(image_files)}")
print(f"Valid landmarks    : {len(valid_ids)}")
print("=" * 70)


# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

processed = 0
success = 0
missing_landmarks = 0
failed = 0


# ------------------------------------------------------------
# Process images
# ------------------------------------------------------------

for index, filename in enumerate(image_files, start=1):

    image_id = os.path.splitext(filename)[0]

    # Only process images with valid MediaPipe landmarks
    if image_id not in valid_ids:
        missing_landmarks += 1
        continue

    image_path = os.path.join(
        FACES_DIR,
        filename
    )

    try:

        image = cv2.imread(image_path)

        if image is None:
            failed += 1
            continue

        # ----------------------------------------------------
        # Face Parsing
        # ----------------------------------------------------

        mask = parser.parse(image)

        if mask is None:
            failed += 1
            continue

        mask = np.asarray(mask)

        if mask.ndim != 2:
            failed += 1
            continue

        # ----------------------------------------------------
        # Save RAW class-ID mask
        # ----------------------------------------------------

        mask_path = os.path.join(
            MASKS_DIR,
            f"{image_id}.png"
        )

        cv2.imwrite(
            mask_path,
            mask.astype(np.uint8)
        )

        # ----------------------------------------------------
        # Create colored visualization
        # ----------------------------------------------------

        colored_mask = colorize_mask(mask)

        face_display = add_title(
            image,
            "Face (512x512)"
        )

        mask_display = add_title(
            colored_mask,
            "Face Parsing Mask"
        )

        combined = np.hstack(
            [
                face_display,
                mask_display
            ]
        )

        # ----------------------------------------------------
        # Save preview
        # ----------------------------------------------------

        preview_path = os.path.join(
            PREVIEW_DIR,
            f"{image_id}_mask.jpg"
        )

        cv2.imwrite(
            preview_path,
            combined,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        processed += 1
        success += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if processed % 100 == 0:

            print(
                f"Processed: {processed} | "
                f"Success: {success} | "
                f"Missing landmarks: {missing_landmarks} | "
                f"Failed: {failed}"
            )

    except Exception as e:

        failed += 1

        print(
            f"FAILED {filename}: {e}"
        )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("FACE PARSING COMPLETED")
print("=" * 70)

print(f"Images found        : {len(image_files)}")
print(f"Valid landmarks     : {len(valid_ids)}")
print(f"Processed           : {processed}")
print(f"Success             : {success}")
print(f"Missing landmarks   : {missing_landmarks}")
print(f"Failed              : {failed}")

print()
print("Masks:")
print(MASKS_DIR)

print()
print("Previews:")
print(PREVIEW_DIR)

print("=" * 70)
```
