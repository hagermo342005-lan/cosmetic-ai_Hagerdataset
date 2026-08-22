from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = Path(
    "processed/faces/CelebAMask-HQ/0.jpg"
)

MASK_DIR = Path(
    "processed/face_parsing_masks"
)

OUTPUT_DIR = Path(
    "processed/cosmetic_results"
)

OUTPUT_PATH = OUTPUT_DIR / "0_nose_slimmed.jpg"

DEBUG_MASK_PATH = OUTPUT_DIR / "0_nose_mask_used.png"


# ------------------------------------------------------------
# Nose width
# 1.00 = no change
# 0.90 = slight slimming
# 0.82 = medium slimming
# 0.75 = strong slimming
# ------------------------------------------------------------

NOSE_WIDTH_SCALE = 0.82


# ============================================================
# FIND MASK
# ============================================================

def find_mask(image_path: Path) -> Path:

    image_id = image_path.stem

    possible_masks = [
        MASK_DIR / f"{image_id}.png",
        MASK_DIR / f"{image_id}_mask.png",
        MASK_DIR / f"{image_id}_parsing.png",
    ]

    for path in possible_masks:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No mask found for image {image_id}\n"
        f"Checked:\n"
        + "\n".join(str(p) for p in possible_masks)
    )


# ============================================================
# LOAD IMAGE
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


MASK_PATH = find_mask(IMAGE_PATH)

print("=" * 70)
print("COSMETIC NOSE SIMULATION")
print("=" * 70)
print(f"Image : {IMAGE_PATH}")
print(f"Mask  : {MASK_PATH}")
print(f"Scale : {NOSE_WIDTH_SCALE}")
print()


image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError(
        f"Could not read image:\n{IMAGE_PATH}"
    )


mask = cv2.imread(
    str(MASK_PATH),
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise RuntimeError(
        f"Could not read mask:\n{MASK_PATH}"
    )


# ============================================================
# RESIZE MASK IF NECESSARY
# ============================================================

if mask.shape[:2] != image.shape[:2]:

    print(
        f"Resizing mask from "
        f"{mask.shape[::-1]} to "
        f"{(image.shape[1], image.shape[0])}"
    )

    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )


# ============================================================
# DETERMINE NOSE REGION
# ============================================================

# Face parsing masks are categorical.
# We don't assume a specific class number blindly.
#
# First inspect the available values.

unique_values = np.unique(mask)

print("Mask unique values:")
print(unique_values)


# ------------------------------------------------------------
# IMPORTANT:
# For the current CelebAMask-HQ parsing setup, the nose
# class must be selected here.
#
# If your preview showed another class ID for the nose,
# change NOSE_CLASS below.
# ------------------------------------------------------------

NOSE_CLASS = 10


nose_mask = np.where(
    mask == NOSE_CLASS,
    255,
    0
).astype(np.uint8)


nose_pixels = int(np.count_nonzero(nose_mask))

print(f"Nose pixels: {nose_pixels}")


if nose_pixels == 0:

    raise RuntimeError(
        "Nose mask is empty.\n"
        "The NOSE_CLASS value is probably incorrect.\n"
        "Check the class IDs from script 27."
    )


# ============================================================
# CLEAN MASK
# ============================================================

kernel = np.ones(
    (5, 5),
    np.uint8
)

nose_mask = cv2.morphologyEx(
    nose_mask,
    cv2.MORPH_CLOSE,
    kernel
)

nose_mask = cv2.morphologyEx(
    nose_mask,
    cv2.MORPH_OPEN,
    kernel
)


# ============================================================
# FIND NOSE BOUNDING BOX
# ============================================================

ys, xs = np.where(nose_mask > 0)

x_min = int(xs.min())
x_max = int(xs.max())

y_min = int(ys.min())
y_max = int(ys.max())

print()
print("Nose bounding box:")
print(f"x: {x_min} -> {x_max}")
print(f"y: {y_min} -> {y_max}")


# ============================================================
# CREATE LOCAL WARP
# ============================================================

height, width = image.shape[:2]

center_x = (x_min + x_max) / 2.0
center_y = (y_min + y_max) / 2.0

nose_width = x_max - x_min + 1
nose_height = y_max - y_min + 1


# ------------------------------------------------------------
# We create a smooth horizontal transformation.
#
# Pixels near the center move toward the center.
# This makes the nose narrower.
# ------------------------------------------------------------

Y, X = np.meshgrid(
    np.arange(height, dtype=np.float32),
    np.arange(width, dtype=np.float32),
    indexing="ij"
)


local_x = X - center_x
local_y = Y - center_y


# Normalized horizontal distance
normalized_x = local_x / max(
    nose_width / 2.0,
    1.0
)


# Smooth influence
influence = np.exp(
    -0.5 * (
        normalized_x ** 2
        +
        (local_y / max(nose_height / 2.0, 1.0)) ** 2
    )
)


# Limit transformation mostly to the nose region
mask_float = nose_mask.astype(
    np.float32
) / 255.0


# Additional blur creates a smooth transition
soft_mask = cv2.GaussianBlur(
    mask_float,
    (0, 0),
    sigmaX=8,
    sigmaY=8
)


soft_mask = np.clip(
    soft_mask,
    0.0,
    1.0
)


# ============================================================
# HORIZONTAL WARP
# ============================================================

# For a narrower nose:
#
# source_x = center + local_x / scale
#
# Example:
# scale = 0.82
#
# Output nose becomes approximately 82% of original width.

source_x = (
    center_x
    +
    local_x / NOSE_WIDTH_SCALE
)

source_y = Y.copy()


# Only apply the warp where the nose influence exists.
map_x = (
    X * (1.0 - soft_mask)
    +
    source_x * soft_mask
)

map_y = (
    Y * (1.0 - soft_mask)
    +
    source_y * soft_mask
)


# Keep coordinates valid
map_x = np.clip(
    map_x,
    0,
    width - 1
).astype(np.float32)

map_y = np.clip(
    map_y,
    0,
    height - 1
).astype(np.float32
)


# ============================================================
# APPLY WARP
# ============================================================

warped_image = cv2.remap(
    image,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_REFLECT
)


# ============================================================
# BLEND ORIGINAL + WARPED
# ============================================================

alpha = soft_mask[..., None]

result = (
    image.astype(np.float32)
    * (1.0 - alpha)
    +
    warped_image.astype(np.float32)
    * alpha
)


result = np.clip(
    result,
    0,
    255
).astype(np.uint8)


# ============================================================
# SAVE RESULT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


success = cv2.imwrite(
    str(OUTPUT_PATH),
    result
)

if not success:
    raise RuntimeError(
        f"Could not save output:\n{OUTPUT_PATH}"
    )


# ============================================================
# SAVE DEBUG MASK
# ============================================================

cv2.imwrite(
    str(DEBUG_MASK_PATH),
    nose_mask
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("COSMETIC SIMULATION COMPLETED")
print("=" * 70)
print(f"Original image : {IMAGE_PATH}")
print(f"Nose mask      : {DEBUG_MASK_PATH}")
print(f"Result         : {OUTPUT_PATH}")
print("=" * 70)