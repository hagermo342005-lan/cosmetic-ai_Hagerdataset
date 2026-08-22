from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "raw_datasets" / "FaceSynthetics" / "data"

images = {
    p.stem
    for p in DATA_DIR.glob("*.png")
    if not p.stem.endswith("_seg")
}

landmarks = {
    p.stem.replace("_ldmks", "")
    for p in DATA_DIR.glob("*_ldmks.txt")
}

segments = {
    p.stem.replace("_seg", "")
    for p in DATA_DIR.glob("*_seg.png")
}

all_ids = sorted(images | landmarks | segments)

complete = []
missing = []

for image_id in all_ids:
    missing_parts = []

    if image_id not in images:
        missing_parts.append("image")

    if image_id not in landmarks:
        missing_parts.append("landmarks")

    if image_id not in segments:
        missing_parts.append("segmentation")

    if missing_parts:
        missing.append((image_id, missing_parts))
    else:
        complete.append(image_id)


# Check landmark counts
landmark_valid = 0
landmark_invalid = 0

for image_id in complete:
    file = DATA_DIR / f"{image_id}_ldmks.txt"

    try:
        lines = [
            line.strip()
            for line in file.read_text().splitlines()
            if line.strip()
        ]

        if len(lines) == 70:
            landmark_valid += 1
        else:
            landmark_invalid += 1

    except Exception:
        landmark_invalid += 1


print("=" * 70)
print("FACESYNTHETICS VALIDATION")
print("=" * 70)

print(f"Images              : {len(images)}")
print(f"Segmentation masks  : {len(segments)}")
print(f"Landmark files      : {len(landmarks)}")

print()
print(f"Complete samples    : {len(complete)}")
print(f"Incomplete samples  : {len(missing)}")

print()
print("Landmark validation")
print("-" * 40)
print(f"70 landmarks        : {landmark_valid}")
print(f"Invalid count       : {landmark_invalid}")

print()
print("=" * 70)

if not missing and landmark_invalid == 0:
    print("STATUS: PASS")
    print("All FaceSynthetics samples are complete.")
else:
    print("STATUS: CHECK REQUIRED")

print("=" * 70)