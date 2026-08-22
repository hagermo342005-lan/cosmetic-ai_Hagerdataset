from pathlib import Path
import json
import csv
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_DIRS = [
    "raw_datasets",
    "processed",
    "processed/faces",
    "processed/landmarks",
    "processed/masks",
    "beauty",
    "real_pairs",
    "synthetic_generation",
    "cosmetic_dataset",
    "cosmetic_dataset/train",
    "cosmetic_dataset/val",
    "cosmetic_dataset/test",
    "models",
    "scripts",
    "backend",
    "frontend",
    "evaluation",
    "docs",
]

EXPECTED_FILES = [
    "dataset_inventory.csv",
    "processed/celeba_hq_crop_report.csv",
    "processed/landmark_quality_celeba_hq.csv",
]

EXPECTED_SCRIPTS = [
    "22_extract_ffhq_landmarks.py",
    "23_visualize_mediapipe_landmarks.py",
    "24_landmark_quality_check.py",
    "25_face_parsing.py",
    "26_face_parsing_celeba_hq.py",
    "crop_celeba_hq.py",
]

def count_files(path):
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())

def ext_counts(path):
    c = Counter()
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                c[p.suffix.lower() or "<no_ext>"] += 1
    return c

def first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None

def print_status(ok, label, detail=""):
    mark = "PASS" if ok else "MISSING"
    print(f"[{mark:<7}] {label}" + (f" -> {detail}" if detail else ""))

def main():
    print("=" * 78)
    print("COSMETIC-AI PROJECT AUDIT")
    print("=" * 78)
    print(f"Project root: {ROOT}")
    print()

    # 1) Repository structure
    print("1. REPOSITORY STRUCTURE")
    for rel in EXPECTED_DIRS:
        p = ROOT / rel
        print_status(p.is_dir(), rel)
    print()

    # 2) Required scripts/reports
    print("2. PIPELINE SCRIPTS / REPORTS")
    for rel in EXPECTED_SCRIPTS:
        p = ROOT / "scripts" / rel
        print_status(p.is_file(), f"scripts/{rel}")
    for rel in EXPECTED_FILES:
        p = ROOT / rel
        print_status(p.is_file(), rel)
    print()

    # 3) Raw datasets
    print("3. RAW DATASETS")
    raw = ROOT / "raw_datasets"
    if raw.is_dir():
        children = [p for p in raw.iterdir() if p.is_dir()]
        if not children:
            print("[MISSING ] No dataset folders found in raw_datasets")
        for d in sorted(children):
            n = count_files(d)
            exts = ", ".join(f"{k}:{v}" for k, v in ext_counts(d).most_common())
            print(f"[FOUND   ] {d.name}: {n:,} files | {exts}")
    else:
        print("[MISSING ] raw_datasets does not exist")
    print()

    # 4) FFHQ processed faces
    print("4. FFHQ FACE PREPROCESSING")
    faces = ROOT / "processed" / "faces"
    ffhq_candidates = [
        faces,
        ROOT / "processed" / "faces" / "FFHQ",
        ROOT / "processed" / "faces" / "ffhq",
    ]
    ffhq_path = first_existing([
        faces / "FFHQ",
        faces / "ffhq",
    ])
    if ffhq_path:
        ffhq_count = count_files(ffhq_path)
        print(f"[FOUND   ] FFHQ face folder: {ffhq_path}")
        print(f"          Images/files: {ffhq_count:,}")
    else:
        root_images = [p for p in faces.iterdir() if p.is_file()] if faces.is_dir() else []
        print(f"[FOUND   ] processed/faces root files: {len(root_images):,}")
        print("          NOTE: this folder may contain FFHQ plus dataset-specific subfolders.")

    # Search for common FFHQ report names
    report_candidates = list(ROOT.rglob("*ffhq*report*.csv"))
    if report_candidates:
        for p in report_candidates[:10]:
            print(f"[FOUND   ] FFHQ report: {p.relative_to(ROOT)}")
    else:
        print("[MISSING ] No FFHQ crop/preprocessing report with 'ffhq' and 'report' in its name")
    print()

    # 5) CelebAMask-HQ crop + landmarks
    print("5. CELEBAMASK-HQ -> CROP -> LANDMARKS")
    celeba_faces = ROOT / "processed" / "faces" / "CelebAMask-HQ"
    celeba_landmarks = ROOT / "processed" / "landmarks_celeba_hq"
    celeba_vis = ROOT / "processed" / "landmarks_celeba_hq_colored"

    if celeba_faces.is_dir():
        print(f"[PASS    ] CelebAMask-HQ cropped faces: {count_files(celeba_faces):,}")
    else:
        print("[MISSING ] processed/faces/CelebAMask-HQ")

    if celeba_landmarks.is_dir():
        print(f"[PASS    ] CelebAMask-HQ landmarks: {count_files(celeba_landmarks):,}")
    else:
        print("[MISSING ] processed/landmarks_celeba_hq")

    if celeba_vis.is_dir():
        print(f"[PASS    ] Landmark visualization: {count_files(celeba_vis):,}")
    else:
        print("[MISSING ] processed/landmarks_celeba_hq_colored")

    q = ROOT / "processed" / "landmark_quality_celeba_hq.csv"
    if q.is_file():
        print(f"[PASS    ] Landmark quality report: {q}")
    else:
        print("[MISSING ] Landmark quality report")
    print()

    # 6) Face parsing
    print("6. FACE PARSING / MASKS")
    parsing_model_candidates = [
        ROOT / "models" / "face_parsing" / "parsing_resnet18.onnx",
        ROOT / "models" / "face_parsing" / "model.onnx",
    ]
    model = first_existing(parsing_model_candidates)
    print_status(model is not None, "Face parsing model", str(model.relative_to(ROOT)) if model else "")

    mask_candidates = [
        ROOT / "processed" / "masks",
        ROOT / "processed" / "masks_celeba_hq",
        ROOT / "processed" / "face_parsing_masks",
    ]
    found_mask = None
    for p in mask_candidates:
        if p.is_dir():
            found_mask = p
            break
    print_status(found_mask is not None, "Processed masks folder",
                 f"{found_mask.relative_to(ROOT)} ({count_files(found_mask):,} files)" if found_mask else "")

    preview_candidates = [
        ROOT / "processed" / "masks_preview",
        ROOT / "processed" / "face_parsing_preview",
    ]
    found_preview = first_existing(preview_candidates)
    print_status(found_preview is not None, "Mask preview folder",
                 str(found_preview.relative_to(ROOT)) if found_preview else "")
    print()

    # 7) Beauty score
    print("7. BEAUTY SCORE")
    beauty = ROOT / "beauty"
    beauty_model_candidates = list((ROOT / "models").rglob("*beauty*")) if (ROOT / "models").exists() else []
    score_candidates = []
    if beauty.exists():
        score_candidates += list(beauty.rglob("*score*"))
        score_candidates += list(beauty.rglob("*.csv"))
        score_candidates += list(beauty.rglob("*.json"))
    print_status(beauty.is_dir(), "beauty/ directory")
    print_status(bool(beauty_model_candidates), "Beauty model/checkpoint",
                 ", ".join(str(p.relative_to(ROOT)) for p in beauty_model_candidates[:5]))
    print_status(bool(score_candidates), "Beauty data/score artifacts",
                 ", ".join(str(p.relative_to(ROOT)) for p in score_candidates[:5]))
    print()

    # 8) Operations
    print("8. COSMETIC OPERATIONS")
    required_ops = [
        "rhinoplasty",
        "chin_augmentation",
        "jawline_contouring",
        "facelift",
        "blepharoplasty",
        "lip_enhancement",
    ]
    ops_files = []
    for base in [ROOT / "docs", ROOT / "synthetic_generation", ROOT / "cosmetic_dataset"]:
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".json", ".yaml", ".yml", ".csv", ".md", ".py"}:
                    ops_files.append(p)
    op_text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore").lower()
        for p in ops_files[:500]
    )
    for op in required_ops:
        print_status(op in op_text, op)
    print()

    # 9) Real pairs
    print("9. REAL BEFORE/AFTER PAIRS")
    real = ROOT / "real_pairs"
    print_status(real.is_dir(), "real_pairs/")
    if real.is_dir():
        print(f"          Files: {count_files(real):,}")
    print()

    # 10) Synthetic generation
    print("10. SYNTHETIC GENERATION")
    synth = ROOT / "synthetic_generation"
    synth_files = list(synth.rglob("*")) if synth.is_dir() else []
    synth_code = [p for p in synth_files if p.is_file() and p.suffix.lower() == ".py"]
    synth_images = [p for p in synth_files if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
    print_status(synth.is_dir(), "synthetic_generation/")
    print_status(bool(synth_code), "Synthetic generation code",
                 ", ".join(str(p.relative_to(ROOT)) for p in synth_code[:8]))
    print(f"          Synthetic image files: {len(synth_images):,}")
    print()

    # 11) Cosmetic dataset + metadata validation
    print("11. COSMETIC DATASET + METADATA")
    dataset = ROOT / "cosmetic_dataset"
    print_status(dataset.is_dir(), "cosmetic_dataset/")
    total_cases = 0
    metadata_cases = 0
    bad_metadata = []
    for split in ["train", "val", "test"]:
        d = dataset / split
        cases = [p for p in d.iterdir() if p.is_dir()] if d.is_dir() else []
        total_cases += len(cases)
        print(f"          {split}: {len(cases):,} cases")
        for case in cases:
            meta = case / "metadata.json"
            if meta.is_file():
                metadata_cases += 1
                try:
                    obj = json.loads(meta.read_text(encoding="utf-8"))
                    required = ["id","operation","source_type","before","after","landmarks","mask","generator","seed","quality_flag"]
                    missing = [k for k in required if k not in obj]
                    if missing:
                        bad_metadata.append((str(meta.relative_to(ROOT)), missing))
                except Exception as e:
                    bad_metadata.append((str(meta.relative_to(ROOT)), [f"invalid json: {e}"]))
    print(f"          Total cases: {total_cases:,}")
    print(f"          Cases with metadata.json: {metadata_cases:,}")
    if bad_metadata:
        print(f"[MISSING ] Metadata problems: {len(bad_metadata)}")
        for item in bad_metadata[:10]:
            print("          ", item)
    else:
        print("[PASS    ] No metadata schema errors found in scanned cases")
    print()

    # 12) Dependency / environment files
    print("12. ENVIRONMENT / REPRODUCIBILITY")
    for rel in ["requirements.txt", "environment.yml", "pyproject.toml", ".gitignore"]:
        p = ROOT / rel
        print_status(p.is_file(), rel)
    print()

    # 13) Summary
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("This audit is read-only: it does not modify datasets or code.")
    print()
    print("IMPORTANT:")
    print("- A folder existing does NOT prove the corresponding algorithm was actually run.")
    print("- A successful pipeline needs both code + outputs + QC/report evidence.")
    print("- Real and synthetic data must remain distinguishable using metadata source_type.")
    print("- Original files in raw_datasets/ must remain untouched.")
    print()
    print("Recommended order for missing work:")
    print("1) dataset_inventory.csv + dataset sample previews")
    print("2) FFHQ 5,000-image detection/crop/alignment/512x512 pipeline + QC")
    print("3) FFHQ landmarks + grouped nose/eyes/lips/jaw/chin points + QC")
    print("4) FFHQ face parsing/masks + target-region definitions + mask QC")
    print("5) SCUT-FBP5500 beauty-score training/evaluation")
    print("6) Operation configuration for the 6 operations")
    print("7) Real before/after pairs (only with explicit research-use permission)")
    print("8) Synthetic generation with SDXL/ControlNet/Inpainting + candidates + filtering")
    print("9) Metadata + train/val/test split with person-level separation")
    print("10) Final evaluation and website/backend integration")
    print("=" * 78)

if __name__ == "__main__":
    main()
