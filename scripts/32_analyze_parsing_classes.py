from pathlib import Path
from collections import Counter

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MASK_DIR = Path(
    "processed/face_parsing_masks"
)

OUTPUT_FILE = Path(
    "processed/parsing_classes_report.txt"
)


# ============================================================
# FIND MASKS
# ============================================================

mask_files = sorted(
    MASK_DIR.glob("*.png")
)

total_masks = len(mask_files)


print("=" * 70)
print("FACE PARSING CLASS ANALYSIS")
print("=" * 70)

print(f"Mask directory : {MASK_DIR}")
print(f"Total masks    : {total_masks}")
print("=" * 70)
print()


# ============================================================
# COUNTERS
# ============================================================

# Total pixels for each class
pixel_counts = Counter()

# Number of images containing each class
image_counts = Counter()

# First few example images for each class
examples = {}

# All unique classes
all_classes = set()


# ============================================================
# ANALYZE MASKS
# ============================================================

for index, mask_path in enumerate(
    mask_files,
    start=1
):

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        print(
            f"[WARNING] Could not read: "
            f"{mask_path.name}"
        )
        continue


    unique_values, counts = np.unique(
        mask,
        return_counts=True
    )


    image_classes = set(
        int(value)
        for value in unique_values
    )


    all_classes.update(
        image_classes
    )


    # --------------------------------------------------------
    # Count pixels
    # --------------------------------------------------------

    for value, count in zip(
        unique_values,
        counts
    ):

        class_id = int(value)

        pixel_counts[class_id] += int(
            count
        )


    # --------------------------------------------------------
    # Count images containing each class
    # --------------------------------------------------------

    for class_id in image_classes:

        image_counts[class_id] += 1


        # Keep maximum 5 examples
        if class_id not in examples:

            examples[class_id] = []


        if len(examples[class_id]) < 5:

            examples[class_id].append(
                mask_path.name
            )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index % 1000 == 0
        or index == total_masks
    ):

        print(
            f"Processed: "
            f"{index}/{total_masks}"
        )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("CLASSES FOUND")
print("=" * 70)

print(
    f"Number of unique classes: "
    f"{len(all_classes)}"
)

print()


# ============================================================
# TABLE HEADER
# ============================================================

print(
    f"{'Class':<10}"
    f"{'Images':<15}"
    f"{'Pixels':<20}"
    f"{'Example'}"
)

print("-" * 70)


# ============================================================
# CLASS RESULTS
# ============================================================

for class_id in sorted(all_classes):

    example = examples.get(
        class_id,
        ["-"]
    )[0]

    print(
        f"{class_id:<10}"
        f"{image_counts[class_id]:<15}"
        f"{pixel_counts[class_id]:<20}"
        f"{example}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FACE PARSING CLASS ANALYSIS\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        f"Mask directory: {MASK_DIR}\n"
    )

    f.write(
        f"Total masks: {total_masks}\n"
    )

    f.write(
        f"Unique classes: {len(all_classes)}\n"
    )

    f.write("\n")

    f.write(
        f"{'Class':<10}"
        f"{'Images':<15}"
        f"{'Pixels':<20}"
        f"Examples\n"
    )

    f.write(
        "-" * 70 + "\n"
    )


    for class_id in sorted(all_classes):

        example_list = examples.get(
            class_id,
            []
        )

        f.write(
            f"{class_id:<10}"
            f"{image_counts[class_id]:<15}"
            f"{pixel_counts[class_id]:<20}"
            f"{', '.join(example_list)}\n"
        )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"Report saved to:"
)

print(
    OUTPUT_FILE
)

print("=" * 70)