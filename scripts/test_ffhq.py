from pathlib import Path
from PIL import Image

FFHQ_DIR = Path("raw_datasets/FFHQ")

images = list(FFHQ_DIR.rglob("*.png"))

valid = 0
invalid = 0
sizes = {}

for image_path in images:
    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            size = img.size
            sizes[size] = sizes.get(size, 0) + 1

        valid += 1

    except Exception as e:
        invalid += 1
        print(f"INVALID: {image_path.name}")
        print(f"Reason: {e}")

print("=" * 50)
print("FFHQ IMAGE VALIDATION")
print("=" * 50)

print(f"Total PNG images : {len(images)}")
print(f"Valid images     : {valid}")
print(f"Invalid images   : {invalid}")

print("\nImage sizes:")
for size, count in sorted(sizes.items()):
    print(f"{size}: {count}")

print("=" * 50)

if invalid == 0:
    print("RESULT: PASS")
else:
    print("RESULT: FAIL")