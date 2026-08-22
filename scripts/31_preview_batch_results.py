from pathlib import Path
import random

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BEFORE_DIR = Path(
    "processed/faces/CelebAMask-HQ"
)

BEFORE_AFTER_DIR = Path(
    "processed/cosmetic_results_all/before_after"
)

OUTPUT_DIR = Path(
    "processed/cosmetic_results_all/preview"
)

OUTPUT_PATH = OUTPUT_DIR / "50_random_before_after.jpg"

NUM_SAMPLES = 50

# Fixed seed = same 50 images every time
RANDOM_SEED = 42

# Preview layout
COLUMNS = 5
ROWS = 10

THUMB_WIDTH = 320
THUMB_HEIGHT = 320

LABEL_HEIGHT = 35


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND AVAILABLE BEFORE/AFTER RESULTS
# ============================================================

result_files = sorted(
    BEFORE_AFTER_DIR.glob("*_before_after.jpg")
)


print("=" * 70)
print("RANDOM BATCH BEFORE / AFTER PREVIEW")
print("=" * 70)

print(
    f"Available results : {len(result_files)}"
)


if len(result_files) < NUM_SAMPLES:

    raise RuntimeError(
        f"Only {len(result_files)} results found. "
        f"Need at least {NUM_SAMPLES}."
    )


# ============================================================
# RANDOM SAMPLE
# ============================================================

random.seed(RANDOM_SEED)

selected_files = random.sample(
    result_files,
    NUM_SAMPLES
)


print(
    f"Random samples    : {NUM_SAMPLES}"
)

print(
    f"Random seed       : {RANDOM_SEED}"
)

print()


# ============================================================
# CREATE THUMBNAILS
# ============================================================

tiles = []


for index, result_path in enumerate(
    selected_files,
    start=1
):

    image = cv2.imread(
        str(result_path)
    )

    if image is None:

        print(
            f"[{index}/{NUM_SAMPLES}] "
            f"Could not read: "
            f"{result_path.name}"
        )

        continue


    # --------------------------------------------------------
    # Resize while preserving aspect ratio
    # --------------------------------------------------------

    h, w = image.shape[:2]

    scale = min(
        THUMB_WIDTH / w,
        THUMB_HEIGHT / h
    )

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # Create white tile
    # --------------------------------------------------------

    tile = np.full(
        (
            THUMB_HEIGHT + LABEL_HEIGHT,
            THUMB_WIDTH,
            3
        ),
        255,
        dtype=np.uint8
    )


    # Center image inside tile

    x_offset = (
        THUMB_WIDTH - new_w
    ) // 2

    y_offset = (
        THUMB_HEIGHT - new_h
    ) // 2


    tile[
        y_offset:
        y_offset + new_h,
        x_offset:
        x_offset + new_w
    ] = resized


    # --------------------------------------------------------
    # Extract image ID
    # --------------------------------------------------------

    image_id = result_path.stem.replace(
        "_before_after",
        ""
    )


    # --------------------------------------------------------
    # Add ID label
    # --------------------------------------------------------

    cv2.putText(
        tile,
        f"{index:02d}  ID: {image_id}",
        (8, THUMB_HEIGHT + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )


    tiles.append(tile)


    print(
        f"[{index:02d}/{NUM_SAMPLES}] "
        f"{image_id}"
    )


# ============================================================
# CHECK TILES
# ============================================================

if len(tiles) == 0:

    raise RuntimeError(
        "No preview images were created."
    )


# ============================================================
# PAD IF NECESSARY
# ============================================================

tile_height = (
    THUMB_HEIGHT + LABEL_HEIGHT
)

tile_width = THUMB_WIDTH


required_tiles = (
    ROWS * COLUMNS
)


while len(tiles) < required_tiles:

    blank_tile = np.full(
        (
            tile_height,
            tile_width,
            3
        ),
        255,
        dtype=np.uint8
    )

    tiles.append(
        blank_tile
    )


# ============================================================
# BUILD GRID
# ============================================================

grid_rows = []


for row in range(ROWS):

    row_tiles = tiles[
        row * COLUMNS:
        (row + 1) * COLUMNS
    ]

    row_image = np.hstack(
        row_tiles
    )

    grid_rows.append(
        row_image
    )


preview = np.vstack(
    grid_rows
)


# ============================================================
# ADD TITLE
# ============================================================

TITLE_HEIGHT = 60

title_area = np.full(
    (
        TITLE_HEIGHT,
        preview.shape[1],
        3
    ),
    255,
    dtype=np.uint8
)


cv2.putText(
    title_area,
    "50 RANDOM BEFORE / AFTER RESULTS",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 0),
    2,
    cv2.LINE_AA
)


preview = np.vstack(
    [
        title_area,
        preview
    ]
)


# ============================================================
# SAVE PREVIEW
# ============================================================

success = cv2.imwrite(
    str(OUTPUT_PATH),
    preview,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        95
    ]
)


if not success:

    raise RuntimeError(
        f"Could not save preview:\n{OUTPUT_PATH}"
    )


# ============================================================
# SAVE SELECTED IMAGE LIST
# ============================================================

LIST_PATH = (
    OUTPUT_DIR
    /
    "50_random_selected.txt"
)


with open(
    LIST_PATH,
    "w",
    encoding="utf-8"
) as f:

    for result_path in selected_files:

        image_id = result_path.stem.replace(
            "_before_after",
            ""
        )

        f.write(
            f"{image_id}\n"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("PREVIEW CREATED")
print("=" * 70)

print(
    f"Results available : {len(result_files)}"
)

print(
    f"Random samples    : {len(selected_files)}"
)

print(
    f"Preview           : {OUTPUT_PATH}"
)

print(
    f"Selected IDs      : {LIST_PATH}"
)

print("=" * 70)