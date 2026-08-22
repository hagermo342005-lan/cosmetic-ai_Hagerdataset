import json
from pathlib import Path

import cv2


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FACES_DIR = ROOT / "processed" / "faces" / "CelebAMask-HQ"

LANDMARKS_DIR = (
    ROOT / "processed" / "landmarks_celeba_hq"
)

OUTPUT_DIR = (
    ROOT / "processed" / "landmarks_celeba_hq_preview"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD LANDMARKS
# ============================================================

def load_landmarks(json_path):

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception:

        return None


    landmarks = data.get(
        "landmarks",
        []
    )


    if not landmarks:

        return None


    points = []


    for p in landmarks:

        if "x" not in p or "y" not in p:

            continue


        points.append(
            (
                float(p["x"]),
                float(p["y"])
            )
        )


    if not points:

        return None


    return points


# ============================================================
# FIND IMAGES
# ============================================================

face_files = sorted(
    [
        p
        for p in FACES_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in [".jpg", ".jpeg", ".png"]
    ]
)


print("=" * 70)
print("CELEBAMASK-HQ LANDMARK VISUALIZATION")
print("=" * 70)

print(f"Faces found    : {len(face_files)}")
print(f"Landmarks dir  : {LANDMARKS_DIR}")
print(f"Output dir     : {OUTPUT_DIR}")

print("=" * 70)


# ============================================================
# COUNTERS
# ============================================================

success = 0
missing_json = 0
failed = 0


# ============================================================
# PROCESS
# ============================================================

for index, image_path in enumerate(
    face_files,
    start=1
):

    image_id = image_path.stem

    json_path = (
        LANDMARKS_DIR
        / f"{image_id}.json"
    )


    # --------------------------------------------------------
    # Missing landmarks
    # --------------------------------------------------------

    if not json_path.exists():

        missing_json += 1

        continue


    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )


        if image is None:

            failed += 1

            continue


        # ----------------------------------------------------
        # Load landmarks
        # ----------------------------------------------------

        landmarks = load_landmarks(
            json_path
        )


        if landmarks is None:

            failed += 1

            continue


        # ----------------------------------------------------
        # Draw landmarks
        # ----------------------------------------------------

        for x, y in landmarks:

            x = int(round(x))
            y = int(round(y))


            if (
                0 <= x < image.shape[1]
                and
                0 <= y < image.shape[0]
            ):

                cv2.circle(
                    image,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )


        # ----------------------------------------------------
        # Save preview
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / image_path.name
        )


        cv2.imwrite(
            str(output_path),
            image
        )


        success += 1


    except Exception as e:

        failed += 1

        print(
            f"FAILED: "
            f"{image_path.name} -> {e}"
        )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 250 == 0:

        print(
            f"Processed: {index}/{len(face_files)} | "
            f"Success: {success} | "
            f"Missing JSON: {missing_json} | "
            f"Failed: {failed}"
        )


# ============================================================
# FINAL
# ============================================================

print()

print("=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)

print(f"Faces found      : {len(face_files)}")
print(f"Visualization OK : {success}")
print(f"Missing JSON     : {missing_json}")
print(f"Failed           : {failed}")

print()

print("Preview folder:")
print(OUTPUT_DIR)

print("=" * 70)