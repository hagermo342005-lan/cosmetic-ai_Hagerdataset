from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

BEFORE_PATH = Path(
    "processed/faces/CelebAMask-HQ/0.jpg"
)

AFTER_PATH = Path(
    "processed/cosmetic_results/0_nose_slimmed.jpg"
)

OUTPUT_PATH = Path(
    "processed/cosmetic_results/0_before_after.jpg"
)


# ============================================================
# LOAD IMAGES
# ============================================================

before = cv2.imread(str(BEFORE_PATH))
after = cv2.imread(str(AFTER_PATH))


if before is None:
    raise RuntimeError(
        f"Could not read BEFORE image:\n{BEFORE_PATH}"
    )


if after is None:
    raise RuntimeError(
        f"Could not read AFTER image:\n{AFTER_PATH}"
    )


# ============================================================
# MAKE SAME SIZE
# ============================================================

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


# ============================================================
# ADD LABELS
# ============================================================

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
    label_height:,
    :
] = before


after_panel[
    label_height:,
    :
] = after


# ============================================================
# TEXT
# ============================================================

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


# ============================================================
# COMBINE SIDE BY SIDE
# ============================================================

comparison = np.hstack(
    [
        before_panel,
        after_panel
    ]
)


# ============================================================
# ADD SEPARATOR
# ============================================================

separator_width = 5

separator = np.full(
    (
        comparison.shape[0],
        separator_width,
        3
    ),
    0,
    dtype=np.uint8
)


comparison = np.hstack(
    [
        before_panel,
        separator,
        after_panel
    ]
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


success = cv2.imwrite(
    str(OUTPUT_PATH),
    comparison,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        95
    ]
)


if not success:
    raise RuntimeError(
        f"Could not save:\n{OUTPUT_PATH}"
    )


# ============================================================
# DONE
# ============================================================

print("=" * 70)
print("BEFORE / AFTER COMPARISON CREATED")
print("=" * 70)
print(f"Before : {BEFORE_PATH}")
print(f"After  : {AFTER_PATH}")
print(f"Output : {OUTPUT_PATH}")
print("=" * 70)