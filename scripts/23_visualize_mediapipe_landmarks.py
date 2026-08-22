import os
import json
import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================

FACES_DIR = r".\processed\faces"
LANDMARKS_DIR = r".\processed\landmarks_ffhq"
OUTPUT_DIR = r".\processed\landmarks_ffhq_preview_all"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def load_landmarks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    landmarks = data.get("landmarks", [])

    if not landmarks:
        return None

    points = []

    for p in landmarks:
        if "x" not in p or "y" not in p:
            continue

        points.append((float(p["x"]), float(p["y"])))

    if len(points) == 0:
        return None

    return points


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("MEDIAPIPE LANDMARK VISUALIZATION - ALL VALID IMAGES")
print("=" * 70)

face_files = sorted([
    f for f in os.listdir(FACES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

landmark_files = sorted([
    f for f in os.listdir(LANDMARKS_DIR)
    if f.lower().endswith(".json")
])

landmark_set = {
    os.path.splitext(f)[0]
    for f in landmark_files
}

print(f"Faces found     : {len(face_files)}")
print(f"Landmark files  : {len(landmark_files)}")
print(f"Output          : {os.path.abspath(OUTPUT_DIR)}")
print("=" * 70)

processed = 0
success = 0
missing_json = 0
failed = 0

for index, image_name in enumerate(face_files, start=1):

    image_id = os.path.splitext(image_name)[0]

    # Only process images that have valid MediaPipe JSON
    if image_id not in landmark_set:
        missing_json += 1
        continue

    image_path = os.path.join(FACES_DIR, image_name)
    json_path = os.path.join(LANDMARKS_DIR, image_id + ".json")

    try:
        image = cv2.imread(image_path)

        if image is None:
            failed += 1
            continue

        landmarks = load_landmarks(json_path)

        if landmarks is None:
            failed += 1
            continue

        # Draw every landmark
        for x, y in landmarks:

            x = int(round(x))
            y = int(round(y))

            # Keep points inside image
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(
                    image,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )

        # Save visualization
        output_path = os.path.join(
            OUTPUT_DIR,
            image_id + "_landmarks.jpg"
        )

        cv2.imwrite(output_path, image)

        success += 1
        processed += 1

        # Progress every 100 images
        if processed % 100 == 0:
            print(
                f"Processed: {processed} | "
                f"Success: {success} | "
                f"Missing JSON: {missing_json} | "
                f"Failed: {failed}"
            )

    except Exception as e:
        failed += 1
        print(f"Failed: {image_name} -> {e}")


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)

print(f"Faces found       : {len(face_files)}")
print(f"Landmark files    : {len(landmark_files)}")
print(f"Processed         : {processed}")
print(f"Visualization OK  : {success}")
print(f"Missing JSON      : {missing_json}")
print(f"Failed            : {failed}")

print()
print("Output:")
print(os.path.abspath(OUTPUT_DIR))

print("=" * 70)