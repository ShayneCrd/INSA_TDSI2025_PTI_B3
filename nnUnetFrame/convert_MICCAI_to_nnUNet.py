#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shutil
import nibabel as nib
from pathlib import Path
NNUNET_ROOT = Path(__file__).resolve().parent


MICCAI_test_path  = NNUNET_ROOT / "MICCAI2016_test/Test/raw"
MICCAI_train_path = NNUNET_ROOT / "MICCAI2016_train/1_Data_registered/0_Data_reg_inter_rigid"

imagesTr_path = NNUNET_ROOT / "nnUNet_raw/Dataset422_TDSI2025/imagesTr"
labelsTr_path = NNUNET_ROOT / "nnUNet_raw/Dataset422_TDSI2025/labelsTr"
imagesTs_path = NNUNET_ROOT / "nnUNet_raw/Dataset422_TDSI2025/imagesTs"

dataset_json_path = NNUNET_ROOT / "nnUNet_raw/Dataset422_TDSI2025/dataset.json"

# nnU-Net modalities mapping (your convention)
# 0000 T1, 0001 T2, 0002 FLAIR
MODALITIES = {
    "0000": "T1",
    "0001": "T2",
    "0002": "FLAIR",
}

# Train filenames (inside each case folder)
TRAIN_MODALITY_FILES = {
    "3DT1": "0000",
    "T2": "0001",
    "3DFLAIR": "0002",
}
TRAIN_LABEL_NAMES = {"Consensus"}  # Consensus.nii or Consensus.nii.gz

# Test filenames (inside each case folder)
"""
TEST_MODALITY_FILES = {
    "T1_preprocessed": "0000",
    "T2_preprocessed": "0001",
    "FLAIR_preprocessed": "0002",
} 					UNCOMMENT IF PREPROCESSED IMGS USED
"""
TEST_MODALITY_FILES = {
    "3DT1": "0000",
    "T2": "0001",
    "3DFLAIR": "0002",
}
# Output case prefix
CASE_PREFIX = "MICCAI2016"


# ============================
# HELPERS
# ============================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def extract_case_id_5digits(folder_name: str) -> str | None:
    """
    Extract the first 5-digit sequence from a folder name.
    Example: 'Training_00123' -> '00123'
    """
    m = re.search(r"(\d{5})", folder_name)
    return m.group(1) if m else None

def find_file_with_base(case_dir: str, base_name: str) -> str | None:
    """
    Find file by base name regardless of .nii or .nii.gz extension.
    Example base_name='3DT1' matches '3DT1.nii' or '3DT1.nii.gz'
    """
    cand_nii_gz = os.path.join(case_dir, f"{base_name}.nii.gz")
    cand_nii = os.path.join(case_dir, f"{base_name}.nii")
    if os.path.isfile(cand_nii_gz):
        return cand_nii_gz
    if os.path.isfile(cand_nii):
        return cand_nii
    return None

def copy_as_niigz(src_path: str, dst_path_niigz: str) -> None:
    """
    Always write output as .nii.gz.
    If src is already .nii.gz, copy2 is fine.
    If src is .nii, we load+save compressed as .nii.gz.
    """
    if src_path.endswith(".nii.gz"):
        shutil.copy2(src_path, dst_path_niigz)
        return

    # src is likely .nii -> compress
    img = nib.load(src_path)
    nib.save(img, dst_path_niigz)

def build_dataset_json(num_training: int) -> None:
    """
    nnU-Net v2 dataset.json
    """
    ds = {
        "name": "TDSI2025_MICCAI2016",
        "description": "MICCAI 2016 MS lesion segmentation dataset converted to nnU-Net format (Dataset422_TDSI2025).",
        "reference": "MICCAI 2016 MS lesion segmentation challenge",
        "licence": "unknown",
        "release": "1.0",
        "numTraining": int(num_training),
        "numTest": 0,  # optional in nnU-Net v2; we can leave 0 safely
        "file_ending": ".nii.gz",
        "channel_names": {
            "0": MODALITIES["0000"],
            "1": MODALITIES["0001"],
            "2": MODALITIES["0002"],
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
    }

    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=2)

    print(f"[OK] dataset.json written: {dataset_json_path}")


# ============================
# BUILDERS
# ============================

def build_labelsTr():
    ensure_dir(labelsTr_path)
    missing = []

    case_dirs = sorted([d for d in os.listdir(MICCAI_train_path)
                        if os.path.isdir(os.path.join(MICCAI_train_path, d))])

    for case_folder in case_dirs:
        case_dir = os.path.join(MICCAI_train_path, case_folder)
        case_id = extract_case_id_5digits(case_folder)

        if case_id is None:
            missing.append((case_folder, "no 5-digit id found"))
            continue

        # find Consensus
        consensus_path = None
        for base in TRAIN_LABEL_NAMES:
            consensus_path = find_file_with_base(case_dir, base)
            if consensus_path:
                break

        if consensus_path is None:
            missing.append((case_folder, "Consensus label not found"))
            continue

        out_name = f"{CASE_PREFIX}_{case_id}.nii.gz"
        out_path = os.path.join(labelsTr_path, out_name)
        copy_as_niigz(consensus_path, out_path)

    if missing:
        print(f"[WARN] labelsTr: {len(missing)} cases missing. First 10:")
        for x in missing[:10]:
            print("  -", x)
    else:
        print("[OK] labelsTr complete.")


def build_imagesTr():
    ensure_dir(imagesTr_path)
    missing = []

    case_dirs = sorted([d for d in os.listdir(MICCAI_train_path)
                        if os.path.isdir(os.path.join(MICCAI_train_path, d))])

    for case_folder in case_dirs:
        case_dir = os.path.join(MICCAI_train_path, case_folder)
        case_id = extract_case_id_5digits(case_folder)

        if case_id is None:
            missing.append((case_folder, "no 5-digit id found"))
            continue

        # For each modality file
        ok_all = True
        for base_name, mod_code in TRAIN_MODALITY_FILES.items():
            src = find_file_with_base(case_dir, base_name)
            if src is None:
                missing.append((case_folder, f"missing {base_name}"))
                ok_all = False
                continue

            out_name = f"{CASE_PREFIX}_{case_id}_{mod_code}.nii.gz"
            out_path = os.path.join(imagesTr_path, out_name)
            copy_as_niigz(src, out_path)

        if not ok_all:
            # continue; we already logged missing
            pass

    if missing:
        print(f"[WARN] imagesTr: {len(missing)} missing entries. First 10:")
        for x in missing[:10]:
            print("  -", x)
    else:
        print("[OK] imagesTr complete.")

def adjust_case_id_for_test(case_id: str, case_folder_name: str) -> str:
    """
    If test case contains '_new_2' or '_newest_2', replace the first digit
    of the 5-digit case_id by '9'.

    Example:
        case_folder = '01047NILE_newest_2'
        case_id = '01047'
        --> '91047'
    """
    if "_new_2" in case_folder_name or "_newest_2" in case_folder_name:
        if len(case_id) == 5 and case_id.isdigit():
            return "9" + case_id[1:]
    return case_id

def resolve_test_case_id(case_folder: str) -> str | None:
    """
    Return final case_id for a MICCAI test case folder,
    or None if no valid 5-digit id is found.
    """
    raw = extract_case_id_5digits(case_folder)
    if raw is None:
        return None
    return adjust_case_id_for_test(raw, case_folder)



def build_imagesTs():
    ensure_dir(imagesTs_path)
    missing = []

    case_dirs = sorted([d for d in os.listdir(MICCAI_test_path)
                        if os.path.isdir(os.path.join(MICCAI_test_path, d))])

    for case_folder in case_dirs:
        case_dir = os.path.join(MICCAI_test_path, case_folder)
        #case_id = extract_case_id_5digits(case_folder)
        case_id = resolve_test_case_id(case_folder)
        if case_id is None:
            missing.append((case_folder, "no 5-digit id found"))
            continue

        ok_all = True
        for base_name, mod_code in TEST_MODALITY_FILES.items():
            src = find_file_with_base(case_dir, base_name)
            if src is None:
                missing.append((case_folder, f"missing {base_name}"))
                ok_all = False
                continue

            out_name = f"{CASE_PREFIX}_{case_id}_{mod_code}.nii.gz"
            out_path = os.path.join(imagesTs_path, out_name)
            copy_as_niigz(src, out_path)

        if not ok_all:
            pass

    if missing:
        print(f"[WARN] imagesTs: {len(missing)} missing entries. First 10:")
        for x in missing[:10]:
            print("  -", x)
    else:
        print("[OK] imagesTs complete.")


def count_training_labels() -> int:
    if not os.path.isdir(labelsTr_path):
        return 0
    return len([f for f in os.listdir(labelsTr_path) if f.endswith(".nii.gz")])


# ============================
# MAIN
# ============================

def main():
    # Ensure target dirs exist (you said they are created, but safe)
    ensure_dir(imagesTr_path)
    ensure_dir(labelsTr_path)
    ensure_dir(imagesTs_path)

    # Build train/test folders
    print("[INFO] Building labelsTr...")
    build_labelsTr()

    print("[INFO] Building imagesTr...")
    build_imagesTr()

    print("[INFO] Building imagesTs...")
    build_imagesTs()

    # dataset.json
    ntr = count_training_labels()
    print(f"[INFO] numTraining (labelsTr count): {ntr}")
    build_dataset_json(num_training=ntr)

    print("\n[DONE] Dataset422_TDSI2025 built.")
    print(f" - imagesTr: {imagesTr_path}")
    print(f" - labelsTr: {labelsTr_path}")
    print(f" - imagesTs: {imagesTs_path}")
    print(f" - dataset.json: {dataset_json_path}")


if __name__ == "__main__":
    main()

