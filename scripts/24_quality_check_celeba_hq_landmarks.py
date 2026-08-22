from pathlib import Path
import json
import csv
import shutil
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FACES_DIR = ROOT / "processed" / "faces"
LANDMARKS_DIR = ROOT / "processed" / "landmarks_ffhq"

REJECTED_DIR = ROOT / "processed" / "faces_rejected_landmarks"
REPORT = ROOT / "processed" / "landmark_quality_report.csv"

REJECTED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD LANDMARKS
# ============================================================

def load_landmarks(json_path):

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    points = None

    if isinstance(data, dict):

        for key in ["landmarks", "points", "face_landmarks"]:

            if key in data:
                points = data[key]
                break

    elif isinstance(data, list):

        points = data

    if points is None:
        return None

    result = []

    for p in points:

        if isinstance(p, dict):

            if "x" in p and "y" in p:
                result.append([
                    float(p["x"]),
                    float(p["y"])
                ])

        elif isinstance(p, (list, tuple)) and len(p) >= 2:

            result.append([
                float(p[0]),
                float(p[1])
            ])

    if not result:
        return None

    return np.asarray(result, dtype=np.float32)


# ============================================================
# MAIN
# ============================================================

images = sorted(
    [
        p for p in FACES_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
)


print("=" * 70)
print("FINAL LANDMARK QUALITY CHECK")
print("=" * 70)

print(f"Faces found     : {len(images)}")
print(f"Landmark folder : {LANDMARKS_DIR}")
print(f"Rejected folder : {REJECTED_DIR}")

print("=" * 70)


rows = []

valid = 0
rejected = 0

reason_counts = {}


# ============================================================
# PROCESS
# ============================================================

for i, image_path in enumerate(images, start=1):

    image_id = image_path.stem

    json_path = LANDMARKS_DIR / f"{image_id}.json"

    reasons = []


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:

        reasons.append("invalid_image")

        width = 512
        height = 512

    else:

        height, width = image.shape[:2]


    # --------------------------------------------------------
    # LANDMARK JSON
    # --------------------------------------------------------

    if not json_path.exists():

        reasons.append("missing_landmarks")

    else:

        points = load_landmarks(json_path)

        if points is None:

            reasons.append("invalid_landmark_file")

        else:

            # ------------------------------------------------
            # Number of landmarks
            # ------------------------------------------------

            if len(points) < 400:

                reasons.append(
                    f"too_few_landmarks_{len(points)}"
                )

            # ------------------------------------------------
            # NaN / Infinity
            # ------------------------------------------------

            if not np.isfinite(points).all():

                reasons.append("invalid_coordinates")

            else:

                # ------------------------------------------------
                # Detect normalized MediaPipe coordinates
                # ------------------------------------------------

                max_x = np.max(points[:, 0])
                max_y = np.max(points[:, 1])

                if max_x <= 1.5 and max_y <= 1.5:

                    points[:, 0] *= width
                    points[:, 1] *= height


                # ------------------------------------------------
                # Bounds
                # ------------------------------------------------

                margin = 3

                x_ok = (
                    (points[:, 0] >= -margin)
                    &
                    (points[:, 0] <= width + margin)
                )

                y_ok = (
                    (points[:, 1] >= -margin)
                    &
                    (points[:, 1] <= height + margin)
                )

                if not np.all(x_ok & y_ok):

                    reasons.append(
                        "landmarks_out_of_bounds"
                    )


                # ------------------------------------------------
                # Basic geometry
                # ------------------------------------------------

                if np.all(x_ok & y_ok):

                    min_x = np.min(points[:, 0])
                    max_x = np.max(points[:, 0])

                    min_y = np.min(points[:, 1])
                    max_y = np.max(points[:, 1])

                    face_width = max_x - min_x
                    face_height = max_y - min_y


                    if face_width < width * 0.20:

                        reasons.append(
                            "landmark_face_too_narrow"
                        )


                    if face_height < height * 0.20:

                        reasons.append(
                            "landmark_face_too_short"
                        )


                    if face_width > width * 0.99:

                        reasons.append(
                            "landmark_face_too_wide"
                        )


                    if face_height > height * 0.99:

                        reasons.append(
                            "landmark_face_too_tall"
                        )


    # ========================================================
    # RESULT
    # ========================================================

    if reasons:

        rejected += 1

        for reason in reasons:

            reason_counts[reason] = (
                reason_counts.get(reason, 0) + 1
            )


        # Move image instead of deleting it

        destination = REJECTED_DIR / image_path.name

        if not destination.exists():

            shutil.move(
                str(image_path),
                str(destination)
            )


        status = "REJECTED"

    else:

        valid += 1

        status = "OK"


    rows.append(
        {
            "id": image_id,
            "image": image_path.name,
            "status": status,
            "reasons": ";".join(reasons),
        }
    )


    if i % 250 == 0:

        print(
            f"Processed: {i} | "
            f"Valid: {valid} | "
            f"Rejected: {rejected}"
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
            "status",
            "reasons"
        ]
    )

    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# FINAL REPORT
# ============================================================

total = len(images)

valid_rate = (
    valid / total * 100
    if total > 0
    else 0
)


print()
print("=" * 70)
print("LANDMARK QUALITY CHECK COMPLETED")
print("=" * 70)

print(f"Original images : {total}")
print(f"Valid images    : {valid}")
print(f"Rejected images : {rejected}")
print(f"Valid rate      : {valid_rate:.2f}%")

print()
print("Rejection reasons:")

if reason_counts:

    for reason, count in sorted(
        reason_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"  {reason:<35}: {count}"
        )

else:

    print("  None")


print()
print("Valid faces:")
print(FACES_DIR)

print()
print("Rejected faces:")
print(REJECTED_DIR)

print()
print("Report:")
print(REPORT)

print("=" * 70)