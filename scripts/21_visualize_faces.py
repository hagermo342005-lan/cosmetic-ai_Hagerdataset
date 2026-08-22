import os
from PIL import Image

INPUT_DIR = r".\processed\faces"
OUTPUT_DIR = r".\processed\faces_preview"

IMAGE_SIZE = 128
COLS = 10
ROWS = 10
PER_PAGE = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print("=" * 70)
print("FFHQ FULL QUALITY CHECK")
print("=" * 70)
print(f"Images found: {len(files)}")

valid = 0
wrong_size = 0
invalid = 0

for index, filename in enumerate(files, start=1):
    path = os.path.join(INPUT_DIR, filename)

    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            if img.size == (512, 512):
                valid += 1
            else:
                wrong_size += 1
                print(f"Wrong size: {filename} -> {img.size}")

    except Exception as e:
        invalid += 1
        print(f"Invalid: {filename} -> {e}")

    if index % 500 == 0:
        print(f"Checked: {index}/{len(files)}")

print()
print("=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)
print(f"Valid 512x512 : {valid}")
print(f"Wrong size    : {wrong_size}")
print(f"Invalid       : {invalid}")
print("=" * 70)

print()
print("Creating preview pages...")

page_count = 0

for start in range(0, len(files), PER_PAGE):

    page_files = files[start:start + PER_PAGE]

    grid = Image.new(
        "RGB",
        (COLS * IMAGE_SIZE, ROWS * IMAGE_SIZE),
        "white"
    )

    for i, filename in enumerate(page_files):

        path = os.path.join(INPUT_DIR, filename)

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

            x = (i % COLS) * IMAGE_SIZE
            y = (i // COLS) * IMAGE_SIZE

            grid.paste(img, (x, y))

        except Exception as e:
            print(f"Preview failed: {filename} -> {e}")

    page_count += 1

    output_path = os.path.join(
        OUTPUT_DIR,
        f"page_{page_count:03d}.jpg"
    )

    grid.save(output_path, "JPEG", quality=95)

    print(f"Created page {page_count}")

print()
print("=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)
print(f"Preview pages : {page_count}")
print(f"Folder        : {os.path.abspath(OUTPUT_DIR)}")
print("=" * 70)
