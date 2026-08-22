import cv2
import json
from pathlib import Path
import mediapipe as mp

# =========================
# Paths
# =========================
INPUT_DIR = Path("processed/faces")
OUTPUT_DIR = Path("processed/landmarks")
MODEL_PATH = Path("models/face_landmarker.task")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# MediaPipe Face Landmarker
# =========================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

landmarker = FaceLandmarker.create_from_options(options)

# =========================
# Process images
# =========================
images = list(INPUT_DIR.glob("*.png")) + list(INPUT_DIR.glob("*.jpg"))

print(f"Found {len(images)} face images.")

success = 0
failed = 0

for image_path in images:

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Cannot read: {image_path}")
        failed += 1
        continue

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print(f"[FAILED] No landmarks: {image_path.name}")
        failed += 1
        continue

    # =========================
    # Save landmarks
    # =========================
    landmarks = []

    for landmark in result.face_landmarks[0]:
        landmarks.append({
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z)
        })

    output_file = OUTPUT_DIR / f"{image_path.stem}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_id": image_path.stem,
                "num_landmarks": len(landmarks),
                "landmarks": landmarks
            },
            f,
            indent=2
        )

    print(f"[OK] {image_path.name} -> {output_file}")

    success += 1

landmarker.close()

print("\n==============================")
print("LANDMARKS COMPLETE")
print("==============================")
print(f"Success : {success}")
print(f"Failed  : {failed}")
print(f"Total   : {success + failed}")