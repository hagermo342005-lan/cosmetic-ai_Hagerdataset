from pathlib import Path
from PIL import Image
import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PARSING_DIR = PROJECT_ROOT / "processed" / "face_parsing_masks"

OUTPUT_DIR = PROJECT_ROOT / "processed" / "target_masks_preview"

# Number of images to preview
MAX_PREVIEW = 20


# ============================================================
# TARGET CLASSES
# ============================================================

TARGET_CLASSES = {
    "nose": 10,
    "mouth": 11,
    "upper_lip": 12,
    "lower_lip": 13,
    "skin": 1,
}


# ============================================================
# CHECK INPUT
# ============================================================

if not PARSING_DIR.exists():
    raise FileNotFoundError(
        f"Parsing masks folder not found:\n{PARSING_DIR}"
    )


parsing_files = sorted(PARSING_DIR.glob("*.png"))

if not parsing_files:
    raise RuntimeError(
        f"No PNG masks found in:\n{PARSING_DIR}"
    )


print("=" * 70)
print("TARGET MASK PREVIEW")
print("=" * 70)

print(f"Parsing directory : {PARSING_DIR}")
print(f"Available masks   : {len(parsing_files):,}")
print(f"Preview images    : {min(MAX_PREVIEW, len(parsing_files))}")
print()


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for target_name in TARGET_CLASSES:
    output_dir = OUTPUT_DIR / target_name
    output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROCESS PREVIEW IMAGES
# ============================================================

preview_files = parsing_files[:MAX_PREVIEW]

statistics = {
    target_name: 0
    for target_name in TARGET_CLASSES
}


for index, parsing_path in enumerate(preview_files, start=1):

    try:
        # ----------------------------------------------------
        # Load parsing mask
        # ----------------------------------------------------
        parsing_image = Image.open(parsing_path).convert("L")
        parsing_array = np.array(parsing_image)

        print(
            f"[{index:02d}/{len(preview_files):02d}] "
            f"{parsing_path.name}"
        )

        # ----------------------------------------------------
        # Create each target mask
        # ----------------------------------------------------
        for target_name, class_id in TARGET_CLASSES.items():

            binary_mask = np.where(
                parsing_array == class_id,
                255,
                0
            ).astype(np.uint8)

            output_path = (
                OUTPUT_DIR
                / target_name
                / parsing_path.name
            )

            Image.fromarray(
                binary_mask,
                mode="L"
            ).save(output_path)

            # Check whether target exists
            if np.any(binary_mask == 255):
                statistics[target_name] += 1

    except Exception as e:

        print(
            f"  ERROR processing {parsing_path.name}: {e}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("TARGET MASK PREVIEW COMPLETED")
print("=" * 70)

print(f"Input parsing masks : {len(parsing_files):,}")
print(f"Preview generated   : {len(preview_files)}")
print()

print("Targets:")

for target_name, class_id in TARGET_CLASSES.items():

    output_dir = OUTPUT_DIR / target_name

    file_count = len(
        list(output_dir.glob("*.png"))
    )

    print(
        f"  {target_name:12s} "
        f"class={class_id:2d} "
        f"files={file_count:2d} "
        f"non-empty={statistics[target_name]:2d}"
    )

print()
print(f"Output directory:")
print(OUTPUT_DIR)

print("=" * 70)