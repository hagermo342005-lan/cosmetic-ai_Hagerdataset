from pathlib import Path
import cv2
import json
import csv
import shutil

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FACES_DIR = ROOT / "processed" / "faces"
LANDMARKS_DIR = ROOT / "processed" / "landmarks_ffhq"
MASKS_DIR = ROOT / "processed" / "masks"

REJECTED_DIR = ROOT / "processed" / "faces_rejected"
REPORT_FILE = ROOT / "processed" / "faces_quality_report.csv"

REJECTED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

EXPECTED_SIZE = (512, 512)

valid_count = 0
rejected_count = 0

rows = []

print("=" * 70)
print("FFHQ FACE QUALITY CHECK + CLEANUP")
print("=" * 70)

images = sorted(FACES_DIR.glob("*.jpg"))

print(f"Images found: {len(images)}")
print()

# ============================================================
# CHECK EACH IMAGE
# ============================================================

for i, image_path in enumerate(images, 1):

    image_id = image_path.stem

    status = "OK"
    problems = []

    # --------------------------------------------------------
    # 1. Image readable?
    # --------------------------------------------------------

    img = cv2.imread(str(image_path))

    if img is None:
        problems.append("unreadable_image")

    else:

        h, w = img.shape[:2]

        # ----------------------------------------------------
        # 2. Correct size?
        # ----------------------------------------------------

        if (w, h) != EXPECTED_SIZE:
            problems.append(f"wrong_size_{w}x{h}")

        # ----------------------------------------------------
        # 3. Very dark / empty image?
        # ----------------------------------------------------

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mean_value = float(gray.mean())
        std_value = float(gray.std())

        if mean_value < 5:
            problems.append("almost_black")

        if std_value < 3:
            problems.append("almost_uniform")

    # --------------------------------------------------------
    # 4. Landmark exists?
    # --------------------------------------------------------

    landmark_file = LANDMARKS_DIR / f"{image_id}.json"

    if not landmark_file.exists():

        problems.append("missing_landmarks")

    else:

        try:

            with open(landmark_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            landmarks = data.get("landmarks", [])

            if len(landmarks) == 0:
                problems.append("empty_landmarks")

        except Exception:
            problems.append("invalid_landmarks_json")

    # --------------------------------------------------------
    # 5. Mask exists?
    # --------------------------------------------------------

    mask_file = MASKS_DIR / f"{image_id}_mask.png"

    if not mask_file.exists():

        problems.append("missing_mask")

    else:

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            problems.append("invalid_mask")

        else:

            mh, mw = mask.shape[:2]

            if (mw, mh) != EXPECTED_SIZE:
                problems.append(f"wrong_mask_size_{mw}x{mh}")

            if cv2.countNonZero(mask) == 0:
                problems.append("empty_mask")

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if problems:

        status = "REJECT"

        rejected_count += 1

        # Move image out of processed/faces
        destination = REJECTED_DIR / image_path.name

        if destination.exists():
            destination.unlink()

        shutil.move(
            str(image_path),
            str(destination)
        )

    else:

        status = "OK"
        valid_count += 1

    rows.append({
        "id": image_id,
        "status": status,
        "problems": ";".join(problems)
    })

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if i % 100 == 0 or i == len(images):

        print(
            f"Processed: {i}/{len(images)} | "
            f"Valid: {valid_count} | "
            f"Rejected: {rejected_count}"
        )

# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["id", "status", "problems"]
    )

    writer.writeheader()
    writer.writerows(rows)

# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("QUALITY CHECK COMPLETED")
print("=" * 70)

print(f"Original images : {len(images)}")
print(f"Valid images    : {valid_count}")
print(f"Rejected images : {rejected_count}")

if len(images) > 0:

    print(
        f"Valid rate      : "
        f"{valid_count / len(images) * 100:.2f}%"
    )

print()
print("Valid images remain in:")
print(FACES_DIR)

print()
print("Rejected images moved to:")
print(REJECTED_DIR)

print()
print("Report:")
print(REPORT_FILE)

print("=" * 70)