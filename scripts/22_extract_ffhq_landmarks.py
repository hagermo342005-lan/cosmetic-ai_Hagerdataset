import json
from pathlib import Path

import cv2
import mediapipe as mp


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "processed" / "faces"
OUTPUT_DIR = BASE_DIR / "processed" / "landmarks_ffhq"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_files = sorted(
    [
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
)

print("=" * 70)
print("FFHQ MEDIAPIPE FACE LANDMARK EXTRACTION")
print("=" * 70)
print(f"Input : {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Images found: {len(image_files)}")
print("=" * 70)


# ============================================================
# MediaPipe Tasks API
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# MediaPipe Face Landmarker model
MODEL_DIR = BASE_DIR / "models" / "mediapipe"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "face_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"""
MediaPipe Face Landmarker model not found:

{MODEL_PATH}

Download the official face_landmarker.task model and place it here.
"""
    )


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


success = 0
failed = 0
no_face = 0
already_exists = 0


# ============================================================
# PROCESS
# ============================================================

with FaceLandmarker.create_from_options(options) as landmarker:

    for index, image_path in enumerate(image_files, start=1):

        output_path = OUTPUT_DIR / f"{image_path.stem}.json"

        if output_path.exists():
            already_exists += 1
            continue

        try:

            image = cv2.imread(str(image_path))

            if image is None:
                failed += 1
                print(f"FAILED READ: {image_path.name}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )

            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                no_face += 1
                print(f"NO FACE: {image_path.name}")
                continue

            face_landmarks = result.face_landmarks[0]

            height, width = image.shape[:2]

            landmarks = []

            for landmark in face_landmarks:

                landmarks.append(
                    {
                        "x": round(float(landmark.x * width), 3),
                        "y": round(float(landmark.y * height), 3),
                        "z": round(float(landmark.z), 6),
                    }
                )

            data = {
                "image": image_path.name,
                "image_width": width,
                "image_height": height,
                "num_landmarks": len(landmarks),
                "landmarks": landmarks,
                "source": "FFHQ_processed",
                "model": "MediaPipe Face Landmarker",
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            success += 1

        except Exception as e:

            failed += 1

            print(
                f"FAILED: {image_path.name} -> {e}"
            )

        if index % 100 == 0:

            print(
                f"Processed {index}/{len(image_files)} | "
                f"Success: {success} | "
                f"No face: {no_face} | "
                f"Failed: {failed} | "
                f"Existing: {already_exists}"
            )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FFHQ LANDMARK EXTRACTION COMPLETED")
print("=" * 70)

print(f"Input images       : {len(image_files)}")
print(f"Success            : {success}")
print(f"No face            : {no_face}")
print(f"Failed             : {failed}")
print(f"Already existed    : {already_exists}")

print()
print("Output:")
print(OUTPUT_DIR)

print("=" * 70)