from pathlib import Path
import cv2
import csv

# =========================
# Paths
# =========================

INPUT_DIR = Path(
    r"raw_datasets\CelebAMask-HQ\CelebAMask-HQ\CelebA-HQ-img"
)

OUTPUT_DIR = Path(
    r"processed\faces\CelebAMask-HQ"
)

REPORT_FILE = Path(
    r"processed\celeba_hq_crop_report.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

# =========================
# Face Detector
# =========================

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load OpenCV Haar Cascade.")

# =========================
# Get images
# =========================

images = sorted(
    INPUT_DIR.glob("*.jpg"),
    key=lambda p: int(p.stem)
)

print(f"Total images found: {len(images)}")

# =========================
# Processing
# =========================

processed = 0
failed_read = 0
no_face = 0
multiple_faces = 0

rows = []

for index, image_path in enumerate(images, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        failed_read += 1

        rows.append([
            image_path.name,
            "failed_read"
        ])

        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:

        no_face += 1

        rows.append([
            image_path.name,
            "no_face"
        ])

        continue

    if len(faces) > 1:
        multiple_faces += 1

    # Choose largest detected face
    x, y, w, h = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    # =========================
    # Padding
    # =========================

    padding = int(0.25 * max(w, h))

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:

        no_face += 1

        rows.append([
            image_path.name,
            "invalid_crop"
        ])

        continue

    # =========================
    # Resize
    # =========================

    face_crop = cv2.resize(
        face_crop,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    # =========================
    # Save
    # =========================

    output_path = OUTPUT_DIR / f"{image_path.stem}.jpg"

    success = cv2.imwrite(
        str(output_path),
        face_crop,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    if success:

        processed += 1

        status = "success"

    else:

        status = "save_failed"

    rows.append([
        image_path.name,
        status
    ])

    # Progress
    if index % 500 == 0:

        print(
            f"Processed: {index}/{len(images)} | "
            f"Successful: {processed} | "
            f"No face: {no_face}"
        )

# =========================
# Save Report
# =========================

with open(
    REPORT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "image",
        "status"
    ])

    writer.writerows(rows)

# =========================
# Final Report
# =========================

print("\n==============================")
print("CelebAMask-HQ Crop Finished")
print("==============================")

print(f"Total images:       {len(images)}")
print(f"Successful crops:   {processed}")
print(f"Failed to read:     {failed_read}")
print(f"No face detected:   {no_face}")
print(f"Multiple faces:     {multiple_faces}")

print(f"\nOutput:")
print(OUTPUT_DIR)

print(f"\nReport:")
print(REPORT_FILE)