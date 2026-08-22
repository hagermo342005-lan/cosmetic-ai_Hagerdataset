from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path(
    "processed/faces/CelebAMask-HQ"
)

MASK_DIR = Path(
    "processed/face_parsing_masks"
)

OUTPUT_DIR = Path(
    "processed/cosmetic_results_all"
)

NOSE_WIDTH_SCALE = 0.82

NOSE_CLASS = 10


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

BEFORE_AFTER_DIR = OUTPUT_DIR / "before_after"
SLIMMED_DIR = OUTPUT_DIR / "slimmed"
MASK_OUTPUT_DIR = OUTPUT_DIR / "nose_masks"

BEFORE_AFTER_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SLIMMED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MASK_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND MASK
# ============================================================

def find_mask(image_id):

    possible_masks = [
        MASK_DIR / f"{image_id}.png",
        MASK_DIR / f"{image_id}_mask.png",
        MASK_DIR / f"{image_id}_parsing.png",
    ]

    for path in possible_masks:

        if path.exists():
            return path

    return None


# ============================================================
# CREATE BEFORE / AFTER
# ============================================================

def create_before_after(before, after):

    height = min(
        before.shape[0],
        after.shape[0]
    )

    width = min(
        before.shape[1],
        after.shape[1]
    )

    before = cv2.resize(
        before,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    after = cv2.resize(
        after,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    label_height = 60

    before_panel = np.full(
        (height + label_height, width, 3),
        255,
        dtype=np.uint8
    )

    after_panel = np.full(
        (height + label_height, width, 3),
        255,
        dtype=np.uint8
    )

    before_panel[
        label_height:
    ] = before

    after_panel[
        label_height:
    ] = after

    cv2.putText(
        before_panel,
        "BEFORE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        after_panel,
        "AFTER",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    separator = np.zeros(
        (
            height + label_height,
            5,
            3
        ),
        dtype=np.uint8
    )

    comparison = np.hstack(
        [
            before_panel,
            separator,
            after_panel
        ]
    )

    return comparison


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(image_path):

    image_id = image_path.stem

    mask_path = find_mask(image_id)

    if mask_path is None:
        return "missing_mask"

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return "failed_image"

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        return "failed_mask"

    # --------------------------------------------------------
    # Resize mask if necessary
    # --------------------------------------------------------

    if mask.shape[:2] != image.shape[:2]:

        mask = cv2.resize(
            mask,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # --------------------------------------------------------
    # Extract nose
    # --------------------------------------------------------

    nose_mask = np.where(
        mask == NOSE_CLASS,
        255,
        0
    ).astype(np.uint8)

    if np.count_nonzero(nose_mask) == 0:
        return "empty_nose_mask"

    # --------------------------------------------------------
    # Clean mask
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Nose bounding box
    # --------------------------------------------------------

    ys, xs = np.where(
        nose_mask > 0
    )

    if len(xs) == 0:
        return "empty_nose_mask"

    x_min = int(xs.min())
    x_max = int(xs.max())

    y_min = int(ys.min())
    y_max = int(ys.max())

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    height, width = image.shape[:2]

    center_x = (
        x_min + x_max
    ) / 2.0

    center_y = (
        y_min + y_max
    ) / 2.0

    nose_width = (
        x_max - x_min + 1
    )

    nose_height = (
        y_max - y_min + 1
    )

    # --------------------------------------------------------
    # Coordinate grid
    # --------------------------------------------------------

    Y, X = np.meshgrid(
        np.arange(
            height,
            dtype=np.float32
        ),
        np.arange(
            width,
            dtype=np.float32
        ),
        indexing="ij"
    )

    local_x = X - center_x
    local_y = Y - center_y

    normalized_x = (
        local_x
        /
        max(
            nose_width / 2.0,
            1.0
        )
    )

    influence = np.exp(
        -0.5
        *
        (
            normalized_x ** 2
            +
            (
                local_y
                /
                max(
                    nose_height / 2.0,
                    1.0
                )
            ) ** 2
        )
    )

    # --------------------------------------------------------
    # Soft mask
    # --------------------------------------------------------

    mask_float = (
        nose_mask.astype(
            np.float32
        )
        /
        255.0
    )

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

    # --------------------------------------------------------
    # Horizontal warp
    # --------------------------------------------------------

    source_x = (
        center_x
        +
        local_x / NOSE_WIDTH_SCALE
    )

    source_y = Y.copy()

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

    map_x = np.clip(
        map_x,
        0,
        width - 1
    ).astype(np.float32)

    map_y = np.clip(
        map_y,
        0,
        height - 1
    ).astype(np.float32)

    # --------------------------------------------------------
    # Warp
    # --------------------------------------------------------

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # --------------------------------------------------------
    # Blend
    # --------------------------------------------------------

    alpha = soft_mask[..., None]

    result = (
        image.astype(np.float32)
        *
        (1.0 - alpha)
        +
        warped.astype(np.float32)
        *
        alpha
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Save slimmed image
    # --------------------------------------------------------

    slimmed_path = (
        SLIMMED_DIR
        /
        f"{image_id}_nose_slimmed.jpg"
    )

    cv2.imwrite(
        str(slimmed_path),
        result,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    # --------------------------------------------------------
    # Save nose mask
    # --------------------------------------------------------

    mask_output_path = (
        MASK_OUTPUT_DIR
        /
        f"{image_id}_nose_mask.png"
    )

    cv2.imwrite(
        str(mask_output_path),
        nose_mask
    )

    # --------------------------------------------------------
    # Create Before / After
    # --------------------------------------------------------

    comparison = create_before_after(
        image,
        result
    )

    comparison_path = (
        BEFORE_AFTER_DIR
        /
        f"{image_id}_before_after.jpg"
    )

    cv2.imwrite(
        str(comparison_path),
        comparison,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    return "success"


# ============================================================
# MAIN
# ============================================================

images = sorted(
    IMAGE_DIR.glob("*.jpg")
)

total = len(images)

print("=" * 70)
print("BATCH NOSE COSMETIC SIMULATION")
print("=" * 70)

print(f"Images found : {total}")
print(f"Nose class   : {NOSE_CLASS}")
print(f"Scale        : {NOSE_WIDTH_SCALE}")
print(f"Output       : {OUTPUT_DIR}")
print("=" * 70)
print()


success = 0
missing_mask = 0
failed_image = 0
failed_mask = 0
empty_nose = 0


for index, image_path in enumerate(
    images,
    start=1
):

    try:

        status = process_image(
            image_path
        )

    except Exception as e:

        print(
            f"[{index}/{total}] "
            f"{image_path.name} -> ERROR: {e}"
        )

        continue

    if status == "success":

        success += 1

    elif status == "missing_mask":

        missing_mask += 1

    elif status == "failed_image":

        failed_image += 1

    elif status == "failed_mask":

        failed_mask += 1

    elif status == "empty_nose_mask":

        empty_nose += 1

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index % 100 == 0
        or index == total
    ):

        print(
            f"[{index}/{total}] "
            f"Success: {success} | "
            f"Missing mask: {missing_mask} | "
            f"Empty nose: {empty_nose}"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("BATCH PROCESSING COMPLETED")
print("=" * 70)

print(f"Total images       : {total}")
print(f"Successful         : {success}")
print(f"Missing masks      : {missing_mask}")
print(f"Failed images      : {failed_image}")
print(f"Failed masks       : {failed_mask}")
print(f"Empty nose masks   : {empty_nose}")

print()
print("Outputs:")
print(f"Slimmed images     : {SLIMMED_DIR}")
print(f"Nose masks         : {MASK_OUTPUT_DIR}")
print(f"Before/After       : {BEFORE_AFTER_DIR}")

print("=" * 70)