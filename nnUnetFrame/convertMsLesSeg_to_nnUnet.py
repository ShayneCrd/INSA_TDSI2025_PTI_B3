#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# CONFIG (edit if needed)
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # assumes script is inside nnUnetFrame/

# Raw MSLesSeg location (inside repo)
MSLESSEG_ROOT = REPO_ROOT / "nnUnetFrame" / "MSLesSegDataset"

# nnU-Net raw output root
NNUNET_RAW_ROOT = REPO_ROOT / "nnUnetFrame" / "nnUNet_raw"

DATASET_ID = 421
DATASET_SUFFIX = "TDSI2025"
DATASET_DIRNAME = f"Dataset{DATASET_ID:03d}_{DATASET_SUFFIX}"

CASE_PREFIX = "SEP"  # output case prefix in nnU-Net
MODALITIES = ["T1", "FLAIR", "T2"]  # channel order => _0000 _0001 _0002

# Train structure: patient/visit/
TRAIN_IMG_PATTERN = "{patient}_{visit}_{mod}.nii.gz"
TRAIN_MASK_PATTERN = "{patient}_{visit}_MASK.nii.gz"

# Test structure: patient/   (no visits)
TEST_IMG_PATTERN = "{patient}_{mod}.nii.gz"
TEST_MASK_PATTERN = "{patient}_MASK.nii.gz"   # <-- adjust if different

SKIP_INCOMPLETE_CASES = True  # if missing modality/mask, skip case instead of crashing

LABELS = {"background": 0, "lesion": 1}

# ============================================================
# Helpers
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def is_nii(p: Path) -> bool:
    s = str(p)
    return s.endswith(".nii") or s.endswith(".nii.gz")

def list_dirs(p: Path) -> List[Path]:
    if not p.exists():
        return []
    return sorted([x for x in p.iterdir() if x.is_dir()])

def fmt_case_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:03d}"

@dataclass
class Case:
    case_id: str
    split: str          # "train" or "test"
    images: Dict[str, Path]
    mask: Path

# ============================================================
# Discovery
# ============================================================

def discover_train_cases(train_root: Path, start_idx: int) -> Tuple[List[Case], int]:
    """
    Train: MSLesSeg/train/Pxx/Ty/{patient}_{visit}_{mod}.nii.gz + {patient}_{visit}_MASK.nii.gz
    Visits are variable (some patients may not have all T1..T4).
    """
    cases: List[Case] = []
    idx = start_idx

    for patient_dir in list_dirs(train_root):
        patient = patient_dir.name

        for visit_dir in list_dirs(patient_dir):
            visit = visit_dir.name  # e.g. T1, T2, T3, T4

            images: Dict[str, Path] = {}
            missing = False
            for mod in MODALITIES:
                f = visit_dir / TRAIN_IMG_PATTERN.format(patient=patient, visit=visit, mod=mod)
                if not f.exists() or not is_nii(f):
                    missing = True
                    break
                images[mod] = f

            mask = visit_dir / TRAIN_MASK_PATTERN.format(patient=patient, visit=visit)
            if not mask.exists() or not is_nii(mask):
                missing = True

            if missing:
                if SKIP_INCOMPLETE_CASES:
                    continue
                raise FileNotFoundError(f"Missing files for train case {patient}/{visit} in {visit_dir}")

            cases.append(Case(
                case_id=fmt_case_id(CASE_PREFIX, idx),
                split="train",
                images=images,
                mask=mask
            ))
            idx += 1

    return cases, idx


def discover_test_cases(test_root: Path, start_idx: int) -> Tuple[List[Case], int]:
    """
    Test: MSLesSeg/test/Pxx/{patient}_{mod}.nii.gz + {patient}_MASK.nii.gz
    (No visits)
    """
    cases: List[Case] = []
    idx = start_idx

    for patient_dir in list_dirs(test_root):
        patient = patient_dir.name

        images: Dict[str, Path] = {}
        missing = False
        for mod in MODALITIES:
            f = patient_dir / TEST_IMG_PATTERN.format(patient=patient, mod=mod)
            if not f.exists() or not is_nii(f):
                missing = True
                break
            images[mod] = f

        mask = patient_dir / TEST_MASK_PATTERN.format(patient=patient)
        if not mask.exists() or not is_nii(mask):
            missing = True

        if missing:
            if SKIP_INCOMPLETE_CASES:
                continue
            raise FileNotFoundError(f"Missing files for test patient {patient} in {patient_dir}")

        cases.append(Case(
            case_id=fmt_case_id(CASE_PREFIX, idx),
            split="test",
            images=images,
            mask=mask
        ))
        idx += 1

    return cases, idx

# ============================================================
# Writing nnU-Net dataset
# ============================================================

def write_case(case: Case, dataset_dir: Path) -> None:
    imagesTr = dataset_dir / "imagesTr"
    labelsTr = dataset_dir / "labelsTr"
    imagesTs = dataset_dir / "imagesTs"
    labelsTs = dataset_dir / "labelsTs"

    if case.split == "train":
        img_dir, lbl_dir = imagesTr, labelsTr
    else:
        img_dir, lbl_dir = imagesTs, labelsTs

    # images channels
    for chan, mod in enumerate(MODALITIES):
        src = case.images[mod]
        dst = img_dir / f"{case.case_id}_{chan:04d}.nii.gz"
        copy_file(src, dst)

    # mask
    dst_mask = lbl_dir / f"{case.case_id}.nii.gz"
    copy_file(case.mask, dst_mask)


def write_dataset_json(dataset_dir: Path, n_train: int, n_test: int) -> None:
    channel_names = {str(i): MODALITIES[i] for i in range(len(MODALITIES))}
    ds = {
        "name": DATASET_DIRNAME,
        "description": "MSLesSeg converted to nnU-Net format (TDSI2025)",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "channel_names": channel_names,
        "labels": LABELS,
        "numTraining": n_train,
        "numTest": n_test,
        "file_ending": ".nii.gz"
    }
    with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=2)

def main() -> None:
    train_root = MSLESSEG_ROOT / "train"
    test_root = MSLESSEG_ROOT / "test"

    if not train_root.exists():
        raise FileNotFoundError(f"Missing train folder: {train_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"Missing test folder: {test_root}")

    out_dataset_dir = NNUNET_RAW_ROOT / DATASET_DIRNAME

    # Create required nnU-Net folders (including labelsTs)
    ensure_dir(out_dataset_dir / "imagesTr")
    ensure_dir(out_dataset_dir / "labelsTr")
    ensure_dir(out_dataset_dir / "imagesTs")
    ensure_dir(out_dataset_dir / "labelsTs")

    idx0 = 1
    train_cases, next_idx = discover_train_cases(train_root, idx0)
    test_cases, _ = discover_test_cases(test_root, next_idx)

    print(f"[INFO] Train cases written to imagesTr/labelsTr: {len(train_cases)}")
    print(f"[INFO] Test  cases written to imagesTs/labelsTs: {len(test_cases)}")

    for c in train_cases:
        write_case(c, out_dataset_dir)
    for c in test_cases:
        write_case(c, out_dataset_dir)

    write_dataset_json(out_dataset_dir, n_train=len(train_cases), n_test=len(test_cases))

    print(f"[OK] Created nnU-Net dataset at: {out_dataset_dir}")

if __name__ == "__main__":
    main()
