#!/usr/bin/env python3
import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Optional


DATASET_ID = 424
DATASET_NAME = f"Dataset{DATASET_ID:03d}_TDSI2025"

NNUNET_ROOT = Path(__file__).resolve().parent


BASE_ROOT = NNUNET_ROOT / "open_ms_data-master/cross_sectional/coregistered"
TARGET_ROOT = NNUNET_ROOT /  "nnUNet_raw"

TRAIN_FRACTION = 0.80
SEED = 42

COPY_MODE = "copy"      # "copy" or "symlink"
OVERWRITE = False       # True to delete and rebuild Dataset424_TDSI2025

# Modalities mapping (nnU-Net channels)
# 0000 T1W, 0001 T1WKS, 0002 T2W, 0003 FLAIR
MOD_FILES = [
    ("T1W.nii.gz", 0),
    ("T1WKS.nii.gz", 1),
    ("T2W.nii.gz", 2),
    ("FLAIR.nii.gz", 3),
]

# GT label filename(s) if present. If none found, labels will be skipped.
LABEL_CANDIDATES = [
    "gt.nii.gz",
    "mask.nii.gz",
    "lesion_mask.nii.gz",
    "consensus_gt.nii.gz",
    "label.nii.gz",
    "seg.nii.gz",
]

CHANNEL_NAMES = {
    "0": "T1W",
    "1": "T1WKS",
    "2": "T2W",
    "3": "FLAIR",
}



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
    name = folder_name.lower()
    if not name.startswith("patient"):
        return None
    suffix = name.replace("patient", "")
    if suffix.isdigit():
        return int(suffix)
    return None

def case_id(patient_id: int) -> str:
    return f"OPENMS2_{patient_id:04d}"

def find_label_file(patient_dir: Path) -> Optional[Path]:
    for cand in LABEL_CANDIDATES:
        p = patient_dir / cand
        if p.exists():
            return p
    # also accept .nii (non-gz)
    for cand in [c.replace(".nii.gz", ".nii") for c in LABEL_CANDIDATES if c.endswith(".nii.gz")]:
        p = patient_dir / cand
        if p.exists():
            return p
    return None

def validate_patient(patient_dir: Path) -> Tuple[bool, List[str]]:
    missing = []
    for fname, _ in MOD_FILES:
        if not (patient_dir / fname).exists():
            # also accept .nii (non-gz)
            alt = fname.replace(".nii.gz", ".nii")
            if not (patient_dir / alt).exists():
                missing.append(fname)
    return (len(missing) == 0), missing




def convert_openms2_cross_sectional_to_nnunet_424() -> None:
    dataset_dir = TARGET_ROOT / DATASET_NAME
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

    patient_dirs = sorted([p for p in BASE_ROOT.iterdir() if p.is_dir()])

    valid: List[Tuple[int, Path]] = []
    invalid: List[Tuple[str, List[str]]] = []

    for pdir in patient_dirs:
        pid = parse_patient_id(pdir.name)
        if pid is None:
            continue
        ok, missing = validate_patient(pdir)
        if ok:
            valid.append((pid, pdir))
        else:
            invalid.append((pdir.name, missing))

    if not valid:
        raise RuntimeError(f"No valid patient folders found in {BASE_ROOT}")

    random.seed(SEED)
    random.shuffle(valid)

    n_total = len(valid)
    n_train = int(round(TRAIN_FRACTION * n_total))
    n_train = max(1, min(n_train, n_total - 1)) if n_total > 1 else n_total

    train_set = valid[:n_train]
    test_set = valid[n_train:]

    print("============================================================")
    print(f"Building {DATASET_NAME} (OPENMS2)")
    print(f"BASE_ROOT: {BASE_ROOT}")
    print(f"Total valid patients: {n_total}")
    print(f"Train/Test split: {len(train_set)}/{len(test_set)} (seed={SEED}, train_fraction={TRAIN_FRACTION})")
    if invalid:
        print(f"Invalid folders: {len(invalid)} (first 10)")
        for name, miss in invalid[:10]:
            print(f"  - {name}: missing {miss}")
    print("============================================================")

    def write_subset(subset: List[Tuple[int, Path]], out_images: Path, out_labels: Path) -> Tuple[int, int]:
        cases_written = 0
        labels_written = 0

        for pid, pdir in subset:
            cid = case_id(pid)

            # write modalities
            for fname, ch in MOD_FILES:
                src = pdir / fname
                if not src.exists():
                    src = pdir / fname.replace(".nii.gz", ".nii")
                dst = out_images / f"{cid}_{ch:04d}.nii.gz"
                copy_or_symlink(src, dst)

            # label (optional)
            lbl = find_label_file(pdir)
            if lbl is not None:
                dst_lbl = out_labels / f"{cid}.nii.gz"
                copy_or_symlink(lbl, dst_lbl)
                labels_written += 1

            cases_written += 1

        return cases_written, labels_written

    tr_cases, tr_labels = write_subset(train_set, imagesTr, labelsTr)
    ts_cases, ts_labels = write_subset(test_set, imagesTs, labelsTs)

    # dataset.json (nnU-Net v2 required keys)
    dataset_json = {
        "channel_names": CHANNEL_NAMES,
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": tr_labels,   # nnU-Net expects number of training labels
        "file_ending": ".nii.gz"
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print("============================================================")
    print(" Done.")
    print(f"Dataset folder: {dataset_dir}")
    print(f"imagesTr cases: {tr_cases} | labelsTr: {tr_labels}")
    print(f"imagesTs cases: {ts_cases} | labelsTs: {ts_labels}")
    if tr_labels == 0:
        print(" No labels found in training set. nnU-Net training will NOT work without labels.")
        print("       Add GT masks in patient folders or adjust LABEL_CANDIDATES.")
  
if __name__ == "__main__":
    convert_openms2_cross_sectional_to_nnunet_424()

