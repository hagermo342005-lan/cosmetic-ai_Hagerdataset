from pathlib import Path
import struct
import json


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "raw_datasets"
    / "SCUT-FBP5500"
    / "SCUT-FBP5500_v2"
    / "facial landmark"
)

OUTPUT_DIR = PROJECT_ROOT / "processed" / "landmarks"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Convert one .pts file
# =========================

def convert_pts_to_json(pts_path: Path):

    data = pts_path.read_bytes()

    # First 4 bytes = number of landmarks
    num_landmarks = struct.unpack("<I", data[:4])[0]

    # Remaining bytes = float32 coordinates
    coordinate_data = data[4:]

    values = struct.unpack(
        "<" + "f" * (len(coordinate_data) // 4),
        coordinate_data
    )

    points = []

    for i in range(0, len(values), 2):

        x = float(values[i])
        y = float(values[i + 1])

        points.append({
            "x": x,
            "y": y
        })

    # Safety check
    if len(points) != num_landmarks:
        raise ValueError(
            f"Landmark count mismatch in {pts_path.name}: "
            f"expected {num_landmarks}, got {len(points)}"
        )

    # JSON structure
    result = {
        "image": pts_path.stem + ".jpg",
        "num_landmarks": num_landmarks,
        "landmarks": points
    }

    output_path = OUTPUT_DIR / f"{pts_path.stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return output_path


# =========================
# Main
# =========================

def main():

    pts_files = sorted(INPUT_DIR.glob("*.pts"))

    print("=" * 60)
    print("SCUT-FBP5500 Landmark Conversion")
    print("=" * 60)

    print(f"Input directory:")
    print(INPUT_DIR)

    print(f"\nFound .pts files: {len(pts_files)}")

    if not pts_files:
        print("\nERROR: No .pts files found.")
        return

    success = 0
    failed = 0

    for i, pts_file in enumerate(pts_files, start=1):

        try:

            convert_pts_to_json(pts_file)

            success += 1

            if i <= 10 or i % 500 == 0:
                print(
                    f"[{i}/{len(pts_files)}] "
                    f"{pts_file.name} -> OK"
                )

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] {pts_file.name}: {e}"
            )

    print("\n" + "=" * 60)
    print("Conversion completed")
    print("=" * 60)

    print(f"Successful: {success}")
    print(f"Failed:     {failed}")
    print(f"Output:     {OUTPUT_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()