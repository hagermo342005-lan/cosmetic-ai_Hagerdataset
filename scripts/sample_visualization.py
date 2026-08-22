from pathlib import Path
from PIL import Image, ImageDraw
import random


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_datasets"
OUTPUT = ROOT / "processed" / "dataset_samples"

DATASETS = [
    "FFHQ",
    "FaceSynthetics",
    "CelebAMask-HQ",
    "SCUT-FBP5500",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

SAMPLES_PER_DATASET = 5
THUMB_SIZE = (256, 256)


OUTPUT.mkdir(parents=True, exist_ok=True)


for dataset in DATASETS:

    dataset_path = RAW / dataset

    images = [
        p for p in dataset_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.seed(42)
    samples = random.sample(
        images,
        min(SAMPLES_PER_DATASET, len(images))
    )

    dataset_output = OUTPUT / dataset
    dataset_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{dataset}")
    print(f"Images available: {len(images)}")

    for i, image_path in enumerate(samples, 1):

        try:
            with Image.open(image_path) as img:

                img = img.convert("RGB")
                original_size = img.size

                img.thumbnail(THUMB_SIZE)

                canvas = Image.new(
                    "RGB",
                    (300, 320),
                    "white"
                )

                x = (300 - img.width) // 2
                y = 10

                canvas.paste(img, (x, y))

                draw = ImageDraw.Draw(canvas)

                draw.text(
                    (10, 275),
                    f"{dataset}",
                    fill="black"
                )

                draw.text(
                    (10, 292),
                    f"Original: {original_size}",
                    fill="black"
                )

                output_file = (
                    dataset_output /
                    f"sample_{i:02d}.jpg"
                )

                canvas.save(
                    output_file,
                    quality=95
                )

                print(
                    f"  [{i}/5] {image_path.name}"
                )

        except Exception as e:

            print(
                f"  ERROR: {image_path} -> {e}"
            )


print("\n========================================")
print("SAMPLE VISUALIZATION FINISHED")
print("========================================")
print(f"Output: {OUTPUT}")