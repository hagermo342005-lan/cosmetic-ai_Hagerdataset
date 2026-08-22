import json
from pathlib import Path

import cv2


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FACES_DIR = (
    ROOT
    / "processed"
    / "faces"
    / "CelebAMask-HQ"
)

LANDMARKS_DIR = (
    ROOT
    / "processed"
    / "landmarks_celeba_hq"
)

OUTPUT_DIR = (
    ROOT
    / "processed"
    / "landmarks_celeba_hq_colored"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MEDIAPIPE FACE MESH CONNECTIONS
# ============================================================

# Face oval
FACE_OVAL = [
    10, 338, 297, 332, 284, 251,
    389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
    10
]


# Left eye
LEFT_EYE = [
    33, 7, 163, 144, 145, 153,
    154, 155, 133, 173, 157, 158,
    159, 160, 161, 246, 33
]


# Right eye
RIGHT_EYE = [
    362, 382, 381, 380, 374, 373,
    390, 249, 263, 466, 388, 387,
    386, 385, 384, 398, 362
]


# Left eyebrow
LEFT_EYEBROW = [
    70, 63, 105, 66, 107,
    55, 65, 52, 53, 46
]


# Right eyebrow
RIGHT_EYEBROW = [
    300, 293, 334, 296, 336,
    285, 295, 282, 283, 276
]


# Nose bridge
NOSE_BRIDGE = [
    168, 6, 197, 195, 5,
    4, 1, 19, 94
]


# Nose
NOSE = [
    98, 97, 2, 326, 327,
    294, 278, 344, 440, 275,
    1, 19, 94, 129, 49,
    102, 64, 98
]


# Nose tip
NOSE_TIP = [
    1, 2, 4, 5, 45, 275,
    440, 460, 294, 278
]


# Outer lips
OUTER_LIPS = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291,
    308, 324, 318, 402, 317,
    14, 87, 178, 88, 95,
    185, 40, 39, 37, 0,
    267, 269, 270, 409, 291,
    61
]


# Inner lips
INNER_LIPS = [
    78, 95, 88, 178, 87,
    14, 317, 402, 318, 324,
    308, 78
]


# Chin
CHIN = [
    152, 148, 176, 149,
    150, 136, 172, 58,
    132, 93, 234
]


# ============================================================
# COLORS (BGR for OpenCV)
# ============================================================

COLORS = {

    "face": (255, 255, 255),

    "left_eye": (255, 0, 0),

    "right_eye": (0, 0, 255),

    "left_eyebrow": (255, 0, 255),

    "right_eyebrow": (255, 0, 255),

    "nose": (0, 165, 255),

    "nose_tip": (0, 255, 255),

    "outer_lips": (0, 255, 0),

    "inner_lips": (0, 200, 0),

    "chin": (255, 255, 0),
}


# ============================================================
# LANDMARK LOADER
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

        if (
            "x" not in p
            or
            "y" not in p
        ):

            continue

        points.append(
            (
                int(round(float(p["x"]))),
                int(round(float(p["y"])))
            )
        )


    if not points:

        return None


    return points


# ============================================================
# DRAW CONNECTION
# ============================================================

def draw_connection(
    image,
    points,
    indices,
    color,
    thickness=1
):

    valid_indices = [
        i
        for i in indices
        if 0 <= i < len(points)
    ]


    for i in range(
        len(valid_indices) - 1
    ):

        p1 = points[
            valid_indices[i]
        ]

        p2 = points[
            valid_indices[i + 1]
        ]


        cv2.line(
            image,
            p1,
            p2,
            color,
            thickness,
            cv2.LINE_AA
        )


# ============================================================
# DRAW POINTS
# ============================================================

def draw_points(
    image,
    points,
    indices,
    color,
    radius=2
):

    for index in indices:

        if not (
            0 <= index < len(points)
        ):

            continue


        x, y = points[index]


        if (
            0 <= x < image.shape[1]
            and
            0 <= y < image.shape[0]
        ):

            cv2.circle(
                image,
                (x, y),
                radius,
                color,
                -1,
                cv2.LINE_AA
            )


# ============================================================
# DRAW LABEL
# ============================================================

def draw_label(
    image,
    text,
    position
):

    x, y = position

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )


# ============================================================
# FIND IMAGES
# ============================================================

image_files = sorted(
    [
        p
        for p in FACES_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in [".jpg", ".jpeg", ".png"]
    ]
)


print("=" * 70)
print(
    "CELEBAMASK-HQ COLORED LANDMARK VISUALIZATION"
)
print("=" * 70)

print(
    f"Images found : {len(image_files)}"
)

print(
    f"Landmarks    : {LANDMARKS_DIR}"
)

print(
    f"Output       : {OUTPUT_DIR}"
)

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
    image_files,
    start=1
):

    image_id = image_path.stem

    json_path = (
        LANDMARKS_DIR
        / f"{image_id}.json"
    )


    if not json_path.exists():

        missing_json += 1

        continue


    try:

        image = cv2.imread(
            str(image_path)
        )


        if image is None:

            failed += 1

            continue


        points = load_landmarks(
            json_path
        )


        if points is None:

            failed += 1

            continue


        # ====================================================
        # DRAW FACE OVAL
        # ====================================================

        draw_connection(
            image,
            points,
            FACE_OVAL,
            COLORS["face"],
            2
        )


        # ====================================================
        # DRAW LEFT EYE
        # ====================================================

        draw_connection(
            image,
            points,
            LEFT_EYE,
            COLORS["left_eye"],
            2
        )

        draw_points(
            image,
            points,
            LEFT_EYE,
            COLORS["left_eye"],
            2
        )


        # ====================================================
        # DRAW RIGHT EYE
        # ====================================================

        draw_connection(
            image,
            points,
            RIGHT_EYE,
            COLORS["right_eye"],
            2
        )

        draw_points(
            image,
            points,
            RIGHT_EYE,
            COLORS["right_eye"],
            2
        )


        # ====================================================
        # EYEBROWS
        # ====================================================

        draw_connection(
            image,
            points,
            LEFT_EYEBROW,
            COLORS["left_eyebrow"],
            2
        )

        draw_connection(
            image,
            points,
            RIGHT_EYEBROW,
            COLORS["right_eyebrow"],
            2
        )


        # ====================================================
        # NOSE
        # ====================================================

        draw_connection(
            image,
            points,
            NOSE_BRIDGE,
            COLORS["nose"],
            2
        )

        draw_connection(
            image,
            points,
            NOSE,
            COLORS["nose"],
            2
        )

        draw_connection(
            image,
            points,
            NOSE_TIP,
            COLORS["nose_tip"],
            2
        )


        draw_points(
            image,
            points,
            NOSE,
            COLORS["nose"],
            2
        )


        # ====================================================
        # OUTER LIPS
        # ====================================================

        draw_connection(
            image,
            points,
            OUTER_LIPS,
            COLORS["outer_lips"],
            2
        )

        draw_points(
            image,
            points,
            OUTER_LIPS,
            COLORS["outer_lips"],
            2
        )


        # ====================================================
        # INNER LIPS
        # ====================================================

        draw_connection(
            image,
            points,
            INNER_LIPS,
            COLORS["inner_lips"],
            2
        )


        # ====================================================
        # CHIN
        # ====================================================

        draw_connection(
            image,
            points,
            CHIN,
            COLORS["chin"],
            2
        )


        # ====================================================
        # IMPORTANT LANDMARK POINTS
        # ====================================================

        important_points = {

            1: "Nose",

            4: "Nose Tip",

            33: "Left Eye",

            263: "Right Eye",

            61: "Left Lip",

            291: "Right Lip",

            152: "Chin",
        }


        for landmark_index, label in (
            important_points.items()
        ):

            if landmark_index >= len(points):

                continue


            x, y = points[
                landmark_index
            ]


            if (
                0 <= x < image.shape[1]
                and
                0 <= y < image.shape[0]
            ):

                cv2.circle(
                    image,
                    (x, y),
                    4,
                    (255, 255, 255),
                    -1
                )

                draw_label(
                    image,
                    label,
                    (x + 5, y - 5)
                )


        # ====================================================
        # SAVE
        # ====================================================

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


    if index % 250 == 0:

        print(
            f"Processed: {index}/{len(image_files)} | "
            f"Success: {success} | "
            f"Missing JSON: {missing_json} | "
            f"Failed: {failed}"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print()

print("=" * 70)
print(
    "COLORED VISUALIZATION COMPLETED"
)
print("=" * 70)

print(
    f"Images found     : {len(image_files)}"
)

print(
    f"Visualization OK : {success}"
)

print(
    f"Missing JSON     : {missing_json}"
)

print(
    f"Failed           : {failed}"
)

print()

print("Output folder:")
print(OUTPUT_DIR)

print("=" * 70)