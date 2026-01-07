#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import nibabel as nib


# ============================================================
# Helpers
# ============================================================

#this is a test modification for GitHub

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_first_existing(folder: str, candidates: list[str]) -> str | None:
    for name in candidates:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    return None


def find_mask_file_in_folder(folder: str, patterns: list[str]) -> str | None:
    """
    Find a mask file inside folder based on regex patterns (case-insensitive).
    Returns first match (sorted for determinism).
    """
    if not os.path.isdir(folder):
        return None

    files = sorted(os.listdir(folder))
    for f in files:
        for pat in patterns:
            if re.search(pat, f, flags=re.IGNORECASE):
                p = os.path.join(folder, f)
                if os.path.isfile(p) and (p.endswith(".nii") or p.endswith(".nii.gz")):
                    return p
    return None


def load_nii(path: str):
    img = nib.load(path)
    data = img.get_fdata()
    # If 4D with last dim 1, squeeze
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Mask must be 3D. Got shape {data.shape} for {path}")
    return img, data


def save_mask_as_uint8(mask_data: np.ndarray, ref_img: nib.Nifti1Image, out_path: str) -> None:
    """
    Binarize mask >0 -> 1 and save as uint8, preserving affine/header.
    """
    mask_bin = (mask_data > 0).astype(np.uint8)
    out_img = nib.Nifti1Image(mask_bin, ref_img.affine, ref_img.header)
    nib.save(out_img, out_path)


# ============================================================
# Dataset 421: MSLesSeg-style -> nnU-Net labelsTs as SEP_###.nii.gz
# ============================================================

def extract_patient_number(name: str) -> int | None:
    """
    Extract integer patient id from strings like:
      P54, P054, sub-P54, etc.
    Returns int or None.
    """
    m = re.search(r"\bP(\d+)\b", name, flags=re.IGNORECASE)
    if not m:
        # try more permissive
        m = re.search(r"P(\d+)", name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def build_labelsTs_421(mslesseg_test_path: str, nnunet_dataset_path: str, shift: int) -> None:
    """
    Build labelsTs for Dataset421:
    - Source: MSLesSeg test folder (patients P54... etc, with MASK somewhere inside)
    - Target: nnUNet_raw/Dataset421.../labelsTs with naming SEP_###.nii.gz

    Mapping:
      SEP_id = patient_id + shift
      Example: P54 + 40 = SEP_094
    """
    labelsTs = os.path.join(nnunet_dataset_path, "labelsTs")
    ensure_dir(labelsTs)

    # Look for mask files with these patterns (adapt if your filenames differ)
    mask_patterns = [
        r"_MASK\.nii(\.gz)?$",
        r"mask\.nii(\.gz)?$",
        r"consensus\.nii(\.gz)?$",  # just in case
    ]

    patients = sorted([d for d in os.listdir(mslesseg_test_path) if os.path.isdir(os.path.join(mslesseg_test_path, d))])
    if not patients:
        raise FileNotFoundError(f"No patient folders found in: {mslesseg_test_path}")

    missing = []
    written = 0

    for p in patients:
        pdir = os.path.join(mslesseg_test_path, p)
        pid = extract_patient_number(p)
        if pid is None:
            missing.append((p, "cannot extract patient number"))
            continue

        # MSLesSeg test sometimes has no visits; sometimes has files directly in patient folder
        # Try: (a) mask in patient folder, (b) mask in first-level subfolders
        mask_path = find_mask_file_in_folder(pdir, mask_patterns)
        if mask_path is None:
            # search one level deeper
            for sub in sorted(os.listdir(pdir)):
                sdir = os.path.join(pdir, sub)
                if os.path.isdir(sdir):
                    mask_path = find_mask_file_in_folder(sdir, mask_patterns)
                    if mask_path is not None:
                        break

        if mask_path is None:
            missing.append((p, "no MASK found"))
            continue

        sep_id = pid + shift
        out_name = f"SEP_{sep_id:03d}.nii.gz"
        out_path = os.path.join(labelsTs, out_name)

        try:
            ref_img, data = load_nii(mask_path)
            save_mask_as_uint8(data, ref_img, out_path)
            written += 1
        except Exception as e:
            missing.append((p, f"failed to convert: {e}"))
            continue

    print(f"[421] labelsTs created: {labelsTs}")
    print(f"[421] Written masks: {written}")
    if missing:
        print(f"[421] Missing/failed: {len(missing)} (first 10 below)")
        for item in missing[:10]:
            print("   -", item)


# ============================================================
# Dataset 422: MICCAI-style -> nnU-Net labelsTs as MICCAI_#####.nii.gz
# ============================================================

def extract_case_id_5digits(text: str) -> str | None:
    """
    Find first 5 consecutive digits anywhere in the string.
    """
    m = re.search(r"(\d{5})", text)
    return m.group(1) if m else None


def adjust_case_id_for_new_variants(case_id_5: str, case_folder_name: str) -> str:
    """
    Specific rule you requested:
    if folder name ends with _new_2 or _newest_2, replace first digit of id by '9'
    Example: 01047NILE_newest_2 -> 91047
    """
    low = case_folder_name.lower()
    if low.endswith("_new_2") or low.endswith("_newest_2"):
        return "9" + case_id_5[1:]
    return case_id_5


def build_labelsTs_422(miccai_test_path: str, nnunet_dataset_path: str) -> None:
    """
    Build labelsTs for Dataset422:
    - Source: MICCAI test folder (case folders)
    - Target: nnUNet_raw/Dataset422.../labelsTs with naming MICCAI_#####.nii.gz

    Mask file:
    - By default searches for "Consensus.nii(.gz)" OR "*mask*.nii(.gz)" OR "*lesion*.nii(.gz)"
      (adapt patterns if needed)
    """
    labelsTs = os.path.join(nnunet_dataset_path, "labelsTs")
    ensure_dir(labelsTs)

    mask_patterns = [
        r"^Consensus\.nii(\.gz)?$",
        r"mask\.nii(\.gz)?$",
        r"lesion.*\.nii(\.gz)?$",
        r"gt.*\.nii(\.gz)?$",
    ]

    cases = sorted([d for d in os.listdir(miccai_test_path) if os.path.isdir(os.path.join(miccai_test_path, d))])
    if not cases:
        raise FileNotFoundError(f"No case folders found in: {miccai_test_path}")

    missing = []
    written = 0

    for case in cases:
        cdir = os.path.join(miccai_test_path, case)
        raw_id = extract_case_id_5digits(case)
        if raw_id is None:
            missing.append((case, "no 5-digit id found"))
            continue

        case_id = adjust_case_id_for_new_variants(raw_id, case)

        mask_path = find_mask_file_in_folder(cdir, mask_patterns)
        if mask_path is None:
            missing.append((case, "no mask found (Consensus/mask/lesion/gt patterns)"))
            continue

        out_name = f"MICCAI2016_{case_id}.nii.gz"
        out_path = os.path.join(labelsTs, out_name)

        try:
            ref_img, data = load_nii(mask_path)
            save_mask_as_uint8(data, ref_img, out_path)
            written += 1
        except Exception as e:
            missing.append((case, f"failed to convert: {e}"))
            continue

    print(f"[422] labelsTs created: {labelsTs}")
    print(f"[422] Written masks: {written}")
    if missing:
        print(f"[422] Missing/failed: {len(missing)} (first 10 below)")
        for item in missing[:10]:
            print("   -", item)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Create nnU-Net labelsTs from source datasets (421/422).")
    ap.add_argument("--dataset_id", type=int, required=True, choices=[421, 422], help="421=SEP (MSLesSeg), 422=MICCAI")
    ap.add_argument("--src", required=True, help="Source dataset root (MSLesSeg test path OR MICCAI test path)")
    ap.add_argument("--nnunet_dataset", required=True, help="Target nnU-Net raw dataset folder (Dataset421_... or Dataset422_...)")
    ap.add_argument("--shift", type=int, default=40, help="Only for 421: SEP_id = P_id + shift (default=40)")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        raise FileNotFoundError(f"Source folder not found: {args.src}")
    if not os.path.isdir(args.nnunet_dataset):
        raise FileNotFoundError(f"nnU-Net dataset folder not found: {args.nnunet_dataset}")

    if args.dataset_id == 421:
        build_labelsTs_421(args.src, args.nnunet_dataset, shift=args.shift)
    else:
        build_labelsTs_422(args.src, args.nnunet_dataset)

    print("[DONE]")


if __name__ == "__main__":
    main()

