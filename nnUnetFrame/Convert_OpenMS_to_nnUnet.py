#!/usr/bin/env python3
import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Optional

# =============================
# CONFIG
# =============================

DATASET_ID = 423
DATASET_NAME = f"Dataset{DATASET_ID:03d}_TDSI2025"

NNUNET_ROOT = Path(__file__).resolve().parent

NNUNET_RAW_ROOT = NNUNET_ROOT / "nnUNet_raw"
BASE_ROOT = NNUNET_ROOT / "open_ms_data-master/longitudinal/coregistered"

TRAIN_FRACTION = 0.80
SEED = 42

# copy or symlink (symlink is faster + saves disk)
COPY_MODE = "copy"   # "copy" or "symlink"
OVERWRITE = False    # True = delete/rebuild the dataset folder if it exists

# Expected files inside each patientXX folder
FILES_6CH = [
    ("study1_T1W.nii.gz", 0),
    ("study1_T2W.nii.gz", 1),
    ("study1_FLAIR.nii.gz", 2),
    ("study2_T1W.nii.gz", 3),
    ("study2_T2W.nii.gz", 4),
    ("study2_FLAIR.nii.gz", 5),
]
GT_NAME = "gt.nii.gz"
BRAINMASK_NAME = "brainmask.nii.gz"  # not used by default

# Channel names for nnU-Net v2
CHANNEL_NAMES = {
    "0": "T1_t1",
    "1": "T2_t1",
    "2": "FLAIR_t1",
    "3": "T1_t2",
    "4": "T2_t2",
    "5": "FLAIR_t2",
}


# =============================
# HELPERS
# =============================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def copy_or_symlink(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if COPY_MODE == "symlink":
        os.symlink(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def parse_patient_id(folder_name: str) -> Optional[int]:
    """
    Expects folder like 'patient01', 'patient02', ...
    Returns integer patient index (1-based) or None if doesn't match.
    """
    name = folder_name.lower()
    if not name.startswith("patient"):
        return None
    suffix = name.replace("patient", "")
    if suffix.isdigit():
        return int(suffix)
    return None

def case_id(patient_id: int) -> str:
    # OPENMS_0001 etc
    return f"OPENMS_{patient_id:04d}"

def validate_patient_folder(pdir: Path) -> Tuple[bool, List[str]]:
    missing = []
    # check 6 modalities
    for fname, _ in FILES_6CH:
        if not (pdir / fname).exists():
            missing.append(fname)
    # check gt
    if not (pdir / GT_NAME).exists():
        missing.append(GT_NAME)
    ok = (len(missing) == 0)
    return ok, missing


# =============================
# MAIN
# =============================

def convert_openms_longitudinal_to_nnunet_423() -> None:
    dataset_dir = NNUNET_RAW_ROOT / DATASET_NAME
    imagesTr = dataset_dir / "imagesTr"
    labelsTr = dataset_dir / "labelsTr"
    imagesTs = dataset_dir / "imagesTs"
    labelsTs = dataset_dir / "labelsTs"

    if dataset_dir.exists() and OVERWRITE:
        shutil.rmtree(dataset_dir)

    ensure_dir(imagesTr)
    ensure_dir(labelsTr)
    ensure_dir(imagesTs)
    ensure_dir(labelsTs)

    # collect valid patients
    patient_dirs = sorted([p for p in BASE_ROOT.iterdir() if p.is_dir()])

    valid: List[Tuple[int, Path]] = []
    invalid: List[Tuple[str, List[str]]] = []

    for pdir in patient_dirs:
        pid = parse_patient_id(pdir.name)
        if pid is None:
            continue
        ok, missing = validate_patient_folder(pdir)
        if ok:
            valid.append((pid, pdir))
        else:
            invalid.append((pdir.name, missing))

    if len(valid) == 0:
        raise RuntimeError(f"No valid patients found in {BASE_ROOT}. Check filenames and paths.")

    # reproducible split
    random.seed(SEED)
    random.shuffle(valid)

    n_total = len(valid)
    n_train = int(round(TRAIN_FRACTION * n_total))
    n_train = max(1, min(n_train, n_total - 1)) if n_total > 1 else n_total

    train_set = valid[:n_train]
    test_set = valid[n_train:]

    print("============================================================")
    print(f"[INFO] Building {DATASET_NAME}")
    print(f"[INFO] BASE_ROOT: {BASE_ROOT}")
    print(f"[INFO] Total valid patients: {n_total}")
    print(f"[INFO] Train/Test split: {len(train_set)}/{len(test_set)} (seed={SEED}, train_fraction={TRAIN_FRACTION})")
    if invalid:
        print(f"[WARN] Invalid patient folders: {len(invalid)} (first 10 below)")
        for name, miss in invalid[:10]:
            print(f"  - {name}: missing {miss}")
    print("============================================================")

    # write cases
    def write_cases(subset: List[Tuple[int, Path]], out_images: Path, out_labels: Path) -> int:
        written = 0
        for pid, pdir in subset:
            cid = case_id(pid)

            # images: 6 channels
            for fname, ch in FILES_6CH:
                src = pdir / fname
                dst = out_images / f"{cid}_{ch:04d}.nii.gz"
                copy_or_symlink(src, dst)

            # label
            src_gt = pdir / GT_NAME
            dst_gt = out_labels / f"{cid}.nii.gz"
            copy_or_symlink(src_gt, dst_gt)

            written += 1
        return written

    n_written_tr = write_cases(train_set, imagesTr, labelsTr)
    n_written_ts = write_cases(test_set, imagesTs, labelsTs)

    # dataset.json (nnU-Net v2 required keys)
    dataset_json = {
        "channel_names": CHANNEL_NAMES,
        "labels": {
            "background": 0,
            "lesion_change": 1
        },
        "numTraining": n_written_tr,
        "file_ending": ".nii.gz"
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print("============================================================")
    print("[OK] Done.")
    print(f"Dataset folder: {dataset_dir}")
    print(f"imagesTr: {imagesTr}  | cases: {n_written_tr}")
    print(f"labelsTr: {labelsTr}  | labels: {n_written_tr}")
    print(f"imagesTs: {imagesTs}  | cases: {n_written_ts}")
    print(f"labelsTs: {labelsTs}  | labels: {n_written_ts}")
    print(f"dataset.json written with numTraining={n_written_tr}")
    print("============================================================")
    print("\nNext step:")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")
    print("")


if __name__ == "__main__":
    convert_openms_longitudinal_to_nnunet_423()

