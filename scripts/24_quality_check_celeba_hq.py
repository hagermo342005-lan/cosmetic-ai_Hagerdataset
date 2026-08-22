import json
import csv
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FACES_DIR = (
    ROOT
    / "processed"
    / "faces"
    / "CelebAMask-HQ"
)

LANDMARKS_DIR = (
    ROOT
    / "processed"
    / "landmarks_celeba_hq"
)

REPORT = (
    ROOT
    / "processed"
    / "landmark_quality_celeba_hq.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_LANDMARKS = 478

MARGIN = 3

MIN_FACE_WIDTH_RATIO = 0.20
MIN_FACE_HEIGHT_RATIO = 0.20

MAX_FACE_WIDTH_RATIO = 0.99
MAX_FACE_HEIGHT_RATIO = 0.99


# ============================================================
# LOAD LANDMARKS
# ============================================================

def load_landmarks(json_path):

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception:

        return None


    landmarks = data.get(
        "landmarks",
        []
    )


    if not isinstance(
        landmarks,
        list
    ):

        return None


    points = []


    for p in landmarks:

        if not isinstance(
            p,
            dict
        ):

            continue


        if (
            "x" not in p
            or
            "y" not in p
        ):

            continue


        try:

            x = float(p["x"])
            y = float(p["y"])


        except Exception:

            return None


        points.append(
            [x, y]
        )


    if not points:

        return None


    return np.asarray(
        points,
        dtype=np.float32
    )


# ============================================================
# FIND IMAGES
# ============================================================

image_files = sorted(
    [
        p
        for p in FACES_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in [".jpg", ".jpeg", ".png"]
    ]
)


print("=" * 70)
print(
    "CELEBAMASK-HQ LANDMARK QUALITY CHECK"
)
print("=" * 70)

print(
    f"Faces folder      : {FACES_DIR}"
)

print(
    f"Landmarks folder  : {LANDMARKS_DIR}"
)

print(
    f"Images found      : {len(image_files)}"
)

print(
    f"Expected points   : {EXPECTED_LANDMARKS}"
)

print("=" * 70)


# ============================================================
# COUNTERS
# ============================================================

valid = 0
rejected = 0

missing_landmarks = 0
invalid_json = 0
wrong_count = 0
invalid_coordinates = 0
out_of_bounds = 0
too_small = 0
too_large = 0
invalid_images = 0


rows = []


# ============================================================
# PROCESS
# ============================================================

for index, image_path in enumerate(
    image_files,
    start=1
):

    image_id = image_path.stem

    json_path = (
        LANDMARKS_DIR
        / f"{image_id}.json"
    )


    reasons = []


    # ========================================================
    # IMAGE CHECK
    # ========================================================

    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        reasons.append(
            "invalid_image"
        )

        invalid_images += 1

        rows.append(
            {
                "id": image_id,
                "image": image_path.name,
                "landmark_count": 0,
                "status": "REVIEW",
                "reasons": ";".join(reasons),
            }
        )

        rejected += 1

        continue


    height, width = image.shape[:2]


    # ========================================================
    # LANDMARK FILE EXISTENCE
    # ========================================================

    if not json_path.exists():

        reasons.append(
            "missing_landmarks"
        )

        missing_landmarks += 1

        rows.append(
            {
                "id": image_id,
                "image": image_path.name,
                "landmark_count": 0,
                "status": "REVIEW",
                "reasons": ";".join(reasons),
            }
        )

        rejected += 1

        continue


    # ========================================================
    # LOAD LANDMARKS
    # ========================================================

    points = load_landmarks(
        json_path
    )


    if points is None:

        reasons.append(
            "invalid_landmark_file"
        )

        invalid_json += 1

        rows.append(
            {
                "id": image_id,
                "image": image_path.name,
                "landmark_count": 0,
                "status": "REVIEW",
                "reasons": ";".join(reasons),
            }
        )

        rejected += 1

        continue


    landmark_count = len(points)


    # ========================================================
    # NUMBER OF LANDMARKS
    # ========================================================

    if landmark_count != EXPECTED_LANDMARKS:

        reasons.append(
            f"wrong_landmark_count_{landmark_count}"
        )

        wrong_count += 1


    # ========================================================
    # NaN / INF CHECK
    # ========================================================

    if not np.isfinite(points).all():

        reasons.append(
            "invalid_coordinates"
        )

        invalid_coordinates += 1


    # ========================================================
    # COORDINATE RANGE CHECK
    # ========================================================

    if np.isfinite(points).all():

        x = points[:, 0]
        y = points[:, 1]


        x_ok = (
            (x >= -MARGIN)
            &
            (x <= width + MARGIN)
        )


        y_ok = (
            (y >= -MARGIN)
            &
            (y <= height + MARGIN)
        )


        if not np.all(
            x_ok & y_ok
        ):

            reasons.append(
                "landmarks_out_of_bounds"
            )

            out_of_bounds += 1


        # ====================================================
        # FACE GEOMETRY
        # ====================================================

        min_x = np.min(x)
        max_x = np.max(x)

        min_y = np.min(y)
        max_y = np.max(y)


        face_width = (
            max_x - min_x
        )

        face_height = (
            max_y - min_y
        )


        # ----------------------------------------------------
        # Too small
        # ----------------------------------------------------

        if (
            face_width
            <
            width * MIN_FACE_WIDTH_RATIO
        ):

            reasons.append(
                "face_too_narrow"
            )

            too_small += 1


        if (
            face_height
            <
            height * MIN_FACE_HEIGHT_RATIO
        ):

            reasons.append(
                "face_too_short"
            )

            too_small += 1


        # ----------------------------------------------------
        # Too large
        # ----------------------------------------------------

        if (
            face_width
            >
            width * MAX_FACE_WIDTH_RATIO
        ):

            reasons.append(
                "face_too_wide"
            )

            too_large += 1


        if (
            face_height
            >
            height * MAX_FACE_HEIGHT_RATIO
        ):

            reasons.append(
                "face_too_tall"
            )

            too_large += 1


    # ========================================================
    # FINAL STATUS
    # ========================================================

    if reasons:

        status = "REVIEW"

        rejected += 1

    else:

        status = "OK"

        valid += 1


    rows.append(
        {
            "id": image_id,
            "image": image_path.name,
            "landmark_count": landmark_count,
            "status": status,
            "reasons": ";".join(reasons),
        }
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if index % 250 == 0:

        print(
            f"Processed: {index}/{len(image_files)} | "
            f"OK: {valid} | "
            f"Review: {rejected}"
        )


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id",
            "image",
            "landmark_count",
            "status",
            "reasons",
        ]
    )

    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# FINAL REPORT
# ============================================================

total = len(image_files)

valid_rate = (
    valid / total * 100
    if total
    else 0
)


print()

print("=" * 70)
print(
    "LANDMARK QUALITY CHECK COMPLETED"
)
print("=" * 70)

print(
    f"Total images       : {total}"
)

print(
    f"Valid              : {valid}"
)

print(
    f"Review required    : {rejected}"
)

print(
    f"Valid rate         : {valid_rate:.2f}%"
)

print()

print(
    f"Missing landmarks  : {missing_landmarks}"
)

print(
    f"Invalid JSON       : {invalid_json}"
)

print(
    f"Wrong point count  : {wrong_count}"
)

print(
    f"Invalid coordinates: {invalid_coordinates}"
)

print(
    f"Out of bounds      : {out_of_bounds}"
)

print(
    f"Invalid images     : {invalid_images}"
)

print(
    f"Too small          : {too_small}"
)

print(
    f"Too large          : {too_large}"
)

print()

print("Report:")
print(REPORT)

print("=" * 70)