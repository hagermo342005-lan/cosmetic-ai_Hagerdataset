import json
from pathlib import Path

# =========================
# Configuration
# =========================

INPUT_DIR = Path("processed/landmarks")
OUTPUT_DIR = Path("processed/landmarks_512")

OLD_SIZE = 350
NEW_SIZE = 512

SCALE_X = NEW_SIZE / OLD_SIZE
SCALE_Y = NEW_SIZE / OLD_SIZE


def convert_landmarks(data):
    """Convert landmark coordinates from 350x350 to 512x512."""

    landmarks = data.get("landmarks")

    if not isinstance(landmarks, list):
        raise ValueError("Missing or invalid 'landmarks' list")

    converted = []

    for point in landmarks:
        if "x" not in point or "y" not in point:
            raise ValueError("Landmark point is missing x or y")

        new_x = point["x"] * SCALE_X
        new_y = point["y"] * SCALE_Y

        converted.append({
            "x": round(new_x, 2),
            "y": round(new_y, 2)
        })

    data["landmarks"] = converted

    # Add metadata describing the conversion
    data["original_size"] = {
        "width": OLD_SIZE,
        "height": OLD_SIZE
    }

    data["image_size"] = {
        "width": NEW_SIZE,
        "height": NEW_SIZE
    }

    data["coordinate_conversion"] = "350x350 -> 512x512"

    return data


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(INPUT_DIR.glob("*.json"))

    print(f"Found {len(json_files)} JSON files.")
    print(f"Scale factor: {SCALE_X:.6f}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    success = 0
    failed = 0

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            converted_data = convert_landmarks(data)

            output_file = OUTPUT_DIR / json_file.name

            with output_file.open("w", encoding="utf-8") as f:
                json.dump(
                    converted_data,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            success += 1

        except Exception as e:
            failed += 1
            print(f"[ERROR] {json_file.name}: {e}")

    print()
    print("=" * 50)
    print("CONVERSION COMPLETE")
    print("=" * 50)
    print(f"Successful: {success}")
    print(f"Failed:     {failed}")
    print(f"Total:      {len(json_files)}")
    print(f"Output:     {OUTPUT_DIR}")


if __name__ == "__main__":
    main()