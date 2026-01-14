#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
import subprocess
import argparse
from glob import glob
from pathlib import Path



THRESHOLD = "0.40"
USE_STRIPPED = True


FILTER_CASES = None

# Dataset suffix mapping 
DATASET_SUFFIX = {
    421: "TDSI2025",
    422: "TDSI2025",
    423: "TDSI2025",
    424: "TDSI2025",  
}


def get_modality_indices(dataset_id: int):
    if dataset_id == 421:
        return 0, 1  # T1, FLAIR
    elif dataset_id == 422:
        return 0, 2  # T1, FLAIR
    elif dataset_id == 423:
        return 6, 8  # diff_T1, diff_FLAIR
    elif dataset_id == 424:
        return 0, 3  # T1, FLAIR
    else:
        raise ValueError("dataset_id must be 421 to 424")



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def clear_temp_dir(path: str):
    """Remove old files from temp_dir so we don't pick the wrong probmap."""
    if not os.path.isdir(path):
        ensure_dir(path)
        return
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        try:
            if os.path.isfile(fp) or os.path.islink(fp):
                os.remove(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
        except Exception:
            pass

def list_case_ids(images_dir: str, list_modality_idx: int = 0):
    patt = os.path.join(images_dir, f"*_{list_modality_idx:04d}.nii*")
    files = sorted(glob(patt))

    case_ids = []
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"(.+)_\d{4}\.nii(\.gz)?$", base)
        if m:
            case_ids.append(m.group(1))
    return sorted(set(case_ids))

def build_case_paths(images_dir: str, case_id: str, t1_idx: int, flair_idx: int):
    def exists_any(p_noext: str):
        p1 = p_noext + ".nii.gz"
        p2 = p_noext + ".nii"
        if os.path.exists(p1):
            return p1
        if os.path.exists(p2):
            return p2
        return None

    t1 = exists_any(os.path.join(images_dir, f"{case_id}_{t1_idx:04d}"))
    flair = exists_any(os.path.join(images_dir, f"{case_id}_{flair_idx:04d}"))
    return t1, flair

def find_lst_output_mask(output_dir: str):
    """Prefer binary mask: space-flair_seg-lst.nii(.gz) inside case_out_dir."""
    preferred = ["space-flair_seg-lst.nii.gz", "space-flair_seg-lst.nii"]
    for name in preferred:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            return p

    # fallback: newest nifti
    outs = []
    for f in os.listdir(output_dir):
        if f.endswith(".nii") or f.endswith(".nii.gz"):
            outs.append(os.path.join(output_dir, f))
    if not outs:
        return None
    return max(outs, key=os.path.getmtime)

def find_prob(temp_dir: str):
    """
    Find LST probability map in temp_dir:
      sub-X_ses-Y_space-FLAIR_seg-lst_prob.nii(.gz)
    """
    preferred = [
        "sub-X_ses-Y_space-FLAIR_seg-lst_prob.nii.gz",
        "sub-X_ses-Y_space-FLAIR_seg-lst_prob.nii",
    ]
    for name in preferred:
        p = os.path.join(temp_dir, name)
        if os.path.exists(p):
            return p
    return None

def command_builder(case_id: str, images_dir: str, base_out_dir: str, dataset_id: int,
                    threshold: str, use_stripped: bool, temp_dir: str, output_pred_dir: str):
    t1_idx, flair_idx = get_modality_indices(dataset_id)
    t1, flair = build_case_paths(images_dir, case_id, t1_idx, flair_idx)
    if not t1 or not flair:
        return None

    case_out_dir = os.path.join(base_out_dir, case_id)
    ensure_dir(case_out_dir)

    cmd = [
        "lst",
        "--t1", t1,
        "--flair", flair,
        "--output", case_out_dir,
        "--threshold", str(threshold),
        "--probability_map",
        "--temp", temp_dir,
    ]
    if use_stripped:
        cmd.append("--stripped")

    # Final outputs:
    case_pred_mask_path = os.path.join(output_pred_dir, f"{case_id}.nii.gz")
    case_pred_prob_path = os.path.join(output_pred_dir, f"{case_id}_prob.nii.gz")
    return cmd, case_out_dir, case_pred_mask_path, case_pred_prob_path



def resolve_paths(dataset_id: int):
    """
    Assumption:
      - this script is in lstFrame/
      - repo root is the parent directory of lstFrame/
    """
    script_dir = Path(__file__).resolve().parent            # .../lstFrame
    repo_root = script_dir.parent                           # repo root

    suffix = DATASET_SUFFIX.get(dataset_id)
    if suffix is None:
        raise ValueError(f"Missing suffix mapping for dataset_id={dataset_id}")

    nnunet_raw_dataset = repo_root / "nnUnetFrame" / "nnUNet_raw" / f"Dataset{dataset_id}_{suffix}"

    # Outputs stay inside lstFrame/
    output_root = script_dir / f"lst_{dataset_id}_outputs_runs_th{THRESHOLD}"
    output_pred_dir = script_dir / f"lst_{dataset_id}_preds_nnunet_th{THRESHOLD}"

    # temp inside lstFrame/
    temp_dir = script_dir / "temp"

    return nnunet_raw_dataset, output_root, output_pred_dir, temp_dir



def run_lst_on_dataset(dataset_id: int):
    nnunet_raw_dataset, output_root, output_pred_dir, temp_dir = resolve_paths(dataset_id)

    imagesTs = nnunet_raw_dataset / "imagesTs"
    if not imagesTs.is_dir():
        raise FileNotFoundError(f"imagesTs not found: {imagesTs}")

    ensure_dir(str(output_root))
    ensure_dir(str(output_pred_dir))
    ensure_dir(str(temp_dir))

    # listing cases (use modality 0000)
    case_ids = list_case_ids(str(imagesTs), list_modality_idx=0)
    if FILTER_CASES is not None:
        wanted = set(FILTER_CASES)
        case_ids = [c for c in case_ids if c in wanted]

    print(f" Dataset={dataset_id} | imagesTs={imagesTs}")
    print(f" Found {len(case_ids)} cases")
    print(f" Output preds dir: {output_pred_dir}")
    print(f" Temp dir: {temp_dir}")

    missing = []
    failed = []
    produced = 0

    for case_id in case_ids:
        built = command_builder(
            case_id=case_id,
            images_dir=str(imagesTs),
            base_out_dir=str(output_root),
            dataset_id=dataset_id,
            threshold=THRESHOLD,
            use_stripped=USE_STRIPPED,
            temp_dir=str(temp_dir),
            output_pred_dir=str(output_pred_dir),
        )
        if built is None:
            missing.append(case_id)
            continue

        cmd, case_out_dir, case_pred_mask_path, case_pred_prob_path = built

        print(f"\n Running LST on {case_id}")
        clear_temp_dir(str(temp_dir))  # avoid mixing probmaps between cases

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            failed.append((case_id, f"lst failed: {e}"))
            continue

        src_mask = find_lst_output_mask(case_out_dir)
        src_prob = find_prob(temp_dir=str(temp_dir))

        if src_mask is None:
            failed.append((case_id, "no mask produced by LST in case output dir"))
            continue
        if src_prob is None:
            failed.append((case_id, "no probmap produced by LST in temp dir"))
            continue

        shutil.copy2(src_mask, case_pred_mask_path)
        shutil.copy2(src_prob, case_pred_prob_path)

        produced += 1
        print(f" Saved mask > {case_pred_mask_path}   (from {os.path.basename(src_mask)})")
        print(f" Saved prob > {case_pred_prob_path}   (from {os.path.basename(src_prob)})")

    print("\n================ SUMMARY ================")
    print(f"Produced predictions: {produced}")
    if missing:
        print(f"Missing modality files (skipped): {len(missing)} (first 10: {missing[:10]})")
    if failed:
        print(f"Failed cases: {len(failed)} (first 10 below)")
        for x in failed[:10]:
            print("  -", x)

    print(f"\nPredictions folder (for benchmark): {output_pred_dir}")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id", type=int, choices=[421, 422, 423, 424],
                   help="Dataset ID: 421/422/423/424")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_lst_on_dataset(args.dataset_id)

