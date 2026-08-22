import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACES_DIR = PROJECT_ROOT / "processed" / "faces"
LANDMARKS_DIR = PROJECT_ROOT / "processed" / "landmarks_ffhq"

OUTPUT_DIR = PROJECT_ROOT / "processed" / "operation_masks"
PREVIEW_DIR = PROJECT_ROOT / "processed" / "operation_masks_preview"

SAMPLE_LIMIT = 50


# MediaPipe Face Mesh landmark groups

NOSE = [
    1, 2, 4, 5, 6,
    19, 45, 48, 64, 98,
    115, 122, 168, 195, 197,
    209, 217, 278, 281, 294,
    327, 344, 351, 358, 370,
    399, 420, 429, 439, 440,
    456, 460
]

LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

LIPS = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291,
    308, 324, 318, 402, 317,
    14, 87, 178, 88, 95,
    78, 191, 80, 81, 82, 13,
    312, 311, 310, 415, 308
]

JAW = [
    172, 136, 150, 149, 176,
    148, 152,
    377, 400, 378, 379, 365,
    397, 288, 361, 323, 454
]

CHEEKS = [
    50, 101, 118, 119, 120, 123,
    147, 187, 205, 207, 213,
    234, 93, 132, 58,
    280, 330, 347, 346, 345,
    352, 376, 411, 427, 436,
    454
]

CHIN = [
    148, 176, 149, 150, 152,
    377, 400, 378, 379
]


def load_landmarks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["landmarks"]


def points_from_indices(landmarks, indices):
    points = []

    for idx in indices:
        if 0 <= idx < len(landmarks):
            x = int(round(landmarks[idx]["x"]))
            y = int(round(landmarks[idx]["y"]))
            points.append([x, y])

    return np.array(points, dtype=np.int32)


def convex_hull_mask(points, shape, expand=0):
    mask = np.zeros(shape[:2], dtype=np.uint8)

    if len(points) < 3:
        return mask

    hull = cv2.convexHull(points)

    cv2.fillConvexPoly(mask, hull, 255)

    if expand > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (expand * 2 + 1, expand * 2 + 1)
        )

        mask = cv2.dilate(mask, kernel)

    return mask


def make_operation_masks(landmarks, image_shape):

    masks = {}

    masks["rhinoplasty"] = convex_hull_mask(
        points_from_indices(landmarks, NOSE),
        image_shape,
        expand=8
    )

    masks["chin_augmentation"] = convex_hull_mask(
        points_from_indices(landmarks, CHIN),
        image_shape,
        expand=18
    )

    masks["jawline_contouring"] = convex_hull_mask(
        points_from_indices(landmarks, JAW),
        image_shape,
        expand=18
    )

    facelift_points = points_from_indices(
        landmarks,
        CHEEKS + JAW
    )

    masks["facelift"] = convex_hull_mask(
        facelift_points,
        image_shape,
        expand=22
    )

    eye_points = points_from_indices(
        landmarks,
        LEFT_EYE + RIGHT_EYE
    )

    masks["blepharoplasty"] = convex_hull_mask(
        eye_points,
        image_shape,
        expand=10
    )

    masks["lip_enhancement"] = convex_hull_mask(
        points_from_indices(landmarks, LIPS),
        image_shape,
        expand=10
    )

    return masks


def create_preview(image, mask):

    overlay = image.copy()

    colored = np.zeros_like(image)
    colored[:, :, 2] = 255

    alpha = 0.35

    region = mask > 0

    overlay[region] = (
        image[region] * (1 - alpha)
        + colored[region] * alpha
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (255, 255, 255),
        2
    )

    return overlay


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    operations = [
        "rhinoplasty",
        "chin_augmentation",
        "jawline_contouring",
        "facelift",
        "blepharoplasty",
        "lip_enhancement"
    ]

    for operation in operations:
        (OUTPUT_DIR / operation).mkdir(
            parents=True,
            exist_ok=True
        )

        (PREVIEW_DIR / operation).mkdir(
            parents=True,
            exist_ok=True
        )

    face_files = sorted(
        FACES_DIR.glob("*.jpg")
    )[:SAMPLE_LIMIT]

    print("=" * 70)
    print("OPERATION TARGET MASKS")
    print("=" * 70)

    print(f"Input faces : {len(face_files)}")
    print(f"Output      : {OUTPUT_DIR}")
    print(f"Preview     : {PREVIEW_DIR}")
    print()

    successful = 0
    failed = 0

    for face_path in face_files:

        image = cv2.imread(str(face_path))

        if image is None:
            failed += 1
            continue

        landmark_path = LANDMARKS_DIR / (
            face_path.stem + ".json"
        )

        if not landmark_path.exists():
            print(
                f"[SKIP] No landmarks: {face_path.name}"
            )
            failed += 1
            continue

        try:

            landmarks = load_landmarks(
                landmark_path
            )

            if len(landmarks) != 478:
                print(
                    f"[SKIP] Invalid landmarks: "
                    f"{face_path.name} "
                    f"({len(landmarks)})"
                )

                failed += 1
                continue

            masks = make_operation_masks(
                landmarks,
                image.shape
            )

            for operation, mask in masks.items():

                mask_path = (
                    OUTPUT_DIR
                    / operation
                    / f"{face_path.stem}.png"
                )

                cv2.imwrite(
                    str(mask_path),
                    mask
                )

                preview = create_preview(
                    image,
                    mask
                )

                preview_path = (
                    PREVIEW_DIR
                    / operation
                    / f"{face_path.stem}.jpg"
                )

                cv2.imwrite(
                    str(preview_path),
                    preview
                )

            successful += 1

        except Exception as e:

            print(
                f"[ERROR] {face_path.name}: {e}"
            )

            failed += 1

    print()
    print("=" * 70)
    print("MASK GENERATION COMPLETED")
    print("=" * 70)

    print(f"Successful : {successful}")
    print(f"Failed     : {failed}")

    print()
    print(f"Masks saved in:")
    print(OUTPUT_DIR)

    print()
    print(f"Previews saved in:")
    print(PREVIEW_DIR)


if __name__ == "__main__":
    main()