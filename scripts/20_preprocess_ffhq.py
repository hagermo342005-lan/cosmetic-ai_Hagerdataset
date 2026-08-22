import os
import cv2
import numpy as np
import csv
from retinaface import RetinaFace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(
    BASE_DIR,
    "raw_datasets",
    "FFHQ",
    "images"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "processed",
    "faces"
)

REPORT_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "ffhq_preprocess_report.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SIZE = 512
MARGIN = 0.15

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def align_face(image, landmarks):
    """Align face using the two eyes."""

    left_eye = np.array(
        landmarks["left_eye"],
        dtype=np.float32
    )

    right_eye = np.array(
        landmarks["right_eye"],
        dtype=np.float32
    )

    target_left = np.array(
        [180.0, 210.0],
        dtype=np.float32
    )

    target_right = np.array(
        [332.0, 210.0],
        dtype=np.float32
    )

    source = np.array(
        [left_eye, right_eye],
        dtype=np.float32
    )

    target = np.array(
        [target_left, target_right],
        dtype=np.float32
    )

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)

    source_centered = source - source_center
    target_centered = target - target_center

    source_distance = np.linalg.norm(
        source_centered[1] - source_centered[0]
    )

    target_distance = np.linalg.norm(
        target_centered[1] - target_centered[0]
    )

    if source_distance < 1e-6:
        return cv2.resize(
            image,
            (TARGET_SIZE, TARGET_SIZE)
        )

    scale = target_distance / source_distance

    angle = np.arctan2(
        source_centered[1][1] - source_centered[0][1],
        source_centered[1][0] - source_centered[0][0]
    )

    angle = angle * 180.0 / np.pi

    transform = cv2.getRotationMatrix2D(
        tuple(source_center),
        angle,
        scale
    )

    transformed_center = np.dot(
        transform[:, :2],
        source_center
    )

    transform[:, 2] += (
        target_center -
        transformed_center
    )

    aligned = cv2.warpAffine(
        image,
        transform,
        (TARGET_SIZE, TARGET_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return aligned


def crop_face(image, facial_area):
    """Crop face with margin."""

    x1, y1, x2, y2 = facial_area

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    width = x2 - x1
    height = y2 - y1

    mx = int(width * MARGIN)
    my = int(height * MARGIN)

    h, w = image.shape[:2]

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)

    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    return image[y1:y2, x1:x2]


def get_largest_face(detections):
    """Return largest detected face."""

    best_key = None
    best_area = 0

    for key, detection in detections.items():

        x1, y1, x2, y2 = detection["facial_area"]

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        area = width * height

        if area > best_area:
            best_area = area
            best_key = key

    if best_key is None:
        return None

    return detections[best_key]


def process_image(input_path, output_path):

    image = cv2.imread(input_path)

    if image is None:
        return False, "Cannot read image"

    try:
        detections = RetinaFace.detect_faces(
            input_path
        )

    except Exception as e:
        return False, f"Detection error: {e}"

    if not detections:
        return False, "No face detected"

    detection = get_largest_face(detections)

    if detection is None:
        return False, "No valid face"

    facial_area = detection["facial_area"]

    cropped = crop_face(
        image,
        facial_area
    )

    if cropped.size == 0:
        return False, "Empty crop"

    temp_path = output_path + ".tmp.jpg"

    cv2.imwrite(
        temp_path,
        cropped,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    try:
        crop_detections = RetinaFace.detect_faces(
            temp_path
        )
    except Exception:
        crop_detections = None

    try:
        os.remove(temp_path)
    except Exception:
        pass

    if crop_detections:

        crop_detection = get_largest_face(
            crop_detections
        )

        if crop_detection is not None:

            crop_landmarks = (
                crop_detection["landmarks"]
            )

            aligned = align_face(
                cropped,
                crop_landmarks
            )

        else:

            aligned = cv2.resize(
                cropped,
                (TARGET_SIZE, TARGET_SIZE),
                interpolation=cv2.INTER_AREA
            )

    else:

        aligned = cv2.resize(
            cropped,
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_AREA
        )

    success = cv2.imwrite(
        output_path,
        aligned,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    
    if not success:
        return False, "Failed to save image"

    return True, "OK"


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("FFHQ FACE PREPROCESSING - ID SAFE VERSION")
print("=" * 70)

print(f"Input : {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Report: {REPORT_PATH}")
print(f"Size  : {TARGET_SIZE}x{TARGET_SIZE}")

if not os.path.exists(INPUT_DIR):
    print("\nERROR: Input directory does not exist.")
    print(INPUT_DIR)
    raise SystemExit(1)

files = []

for filename in os.listdir(INPUT_DIR):

    ext = os.path.splitext(filename)[1].lower()

    if ext in VALID_EXTENSIONS:
        files.append(filename)

files.sort()

print(f"\nImages found: {len(files):,}")

results = []

success_count = 0
failed_count = 0
skipped_count = 0

for index, filename in enumerate(files, start=1):

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    # IMPORTANT:
    # Use the real input filename ID.
    image_id = os.path.splitext(filename)[0]

    output_name = f"{image_id}.jpg"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    # Existing correct output
    if os.path.exists(output_path):

        skipped_count += 1

        results.append([
            image_id,
            filename,
            output_name,
            "SKIPPED",
            "Output already exists"
        ])

    else:

        success, message = process_image(
            input_path,
            output_path
        )

        if success:

            success_count += 1

            results.append([
                image_id,
                filename,
                output_name,
                "SUCCESS",
                message
            ])

        else:

            failed_count += 1

            results.append([
                image_id,
                filename,
                output_name,
                "FAILED",
                message
            ])

            print(
                f"FAILED {filename}: {message}"
            )

    if (
        index % 100 == 0
        or index == len(files)
    ):

        print(
            f"Processed {index:,}/{len(files):,} | "
            f"New Success: {success_count:,} | "
            f"Skipped: {skipped_count:,} | "
            f"Failed: {failed_count:,}"
        )


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "id",
        "input_file",
        "output_file",
        "status",
        "reason"
    ])

    writer.writerows(results)


print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print(f"Input images : {len(files):,}")
print(f"New success  : {success_count:,}")
print(f"Skipped      : {skipped_count:,}")
print(f"Failed       : {failed_count:,}")

print("\nReport:")
print(REPORT_PATH)

print("=" * 70)


