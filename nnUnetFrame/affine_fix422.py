#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fix_affine_miccai422.py

Resample MICCAI2016 nnUNet_raw test set modalities to a common reference grid (FLAIR by default).

Expected input structure:
  nnUNet_raw/Dataset422_TDSI2025/
    imagesTs/
      MICCAI2016_01047_0000.nii.gz
      MICCAI2016_01047_0001.nii.gz
      MICCAI2016_01047_0002.nii.gz  <-- reference (FLAIR)
      ...
    (optional) labelsTs/
      MICCAI2016_01047.nii.gz

This script:
- For each case, loads reference modality (default _0002 = FLAIR)
- Resamples other modalities onto reference grid using linear interpolation
- Optionally resamples labelsTs onto reference grid using nearest neighbor
- Writes outputs into imagesTs_fixed/ and labelsTs_fixed/ by default (non-destructive)
  or overwrites in-place with --inplace (will create backups folder).

Usage:
  python fix_affine_miccai422.py \
    --dataset_raw /local/scardell/nnUnetFrame/nnUNet_raw/Dataset422_TDSI2025 \
    --ref_mod 2 \
    --fix_labels

If you want overwrite:
  python fix_affine_miccai422.py ... --inplace
"""

import os
import argparse
from glob import glob
from typing import List, Tuple

import numpy as np

try:
    import SimpleITK as sitk
except Exception as e:
    raise RuntimeError(
        "SimpleITK is required for this script.\n"
        "Install it in your env:\n"
        "  pip install SimpleITK\n"
        "or:\n"
        "  conda install -c conda-forge simpleitk\n"
        f"Original import error: {e}"
    )


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_case_ids(images_dir: str, ref_mod: int) -> List[str]:
    """List case IDs by searching for *_{ref_mod:04d}.nii*"""
    patt = os.path.join(images_dir, f"*_{ref_mod:04d}.nii*")
    files = sorted(glob(patt))
    case_ids = []
    for f in files:
        base = os.path.basename(f)
        # Split at _{ref_mod:04d}.nii or _{ref_mod:04d}.nii.gz
        token = f"_{ref_mod:04d}.nii"
        if token in base:
            cid = base.split(token)[0]
            case_ids.append(cid)
    return case_ids


def get_mod_path(images_dir: str, case_id: str, mod: int) -> str:
    p_gz = os.path.join(images_dir, f"{case_id}_{mod:04d}.nii.gz")
    p_ni = os.path.join(images_dir, f"{case_id}_{mod:04d}.nii")
    if os.path.exists(p_gz):
        return p_gz
    if os.path.exists(p_ni):
        return p_ni
    return ""


def get_label_path(labels_dir: str, case_id: str) -> str:
    p_gz = os.path.join(labels_dir, f"{case_id}.nii.gz")
    p_ni = os.path.join(labels_dir, f"{case_id}.nii")
    if os.path.exists(p_gz):
        return p_gz
    if os.path.exists(p_ni):
        return p_ni
    return ""


def sitk_read(path: str) -> sitk.Image:
    img = sitk.ReadImage(path)
    return img


def same_grid(a: sitk.Image, b: sitk.Image, tol: float = 1e-6) -> bool:
    """
    Check if two images have identical grid definitions: size, spacing, origin, direction.
    """
    if list(a.GetSize()) != list(b.GetSize()):
        return False

    def close_list(x, y):
        return all(abs(float(i) - float(j)) <= tol for i, j in zip(x, y))

    if not close_list(a.GetSpacing(), b.GetSpacing()):
        return False
    if not close_list(a.GetOrigin(), b.GetOrigin()):
        return False
    if not close_list(a.GetDirection(), b.GetDirection()):
        return False
    return True


def resample_to_ref(
    moving: sitk.Image,
    ref: sitk.Image,
    is_label: bool,
) -> sitk.Image:
    """
    Resample moving image onto ref image grid.
    - Linear for images
    - Nearest neighbor for labels
    """
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    # Identity transform assumes both images already roughly aligned in physical space.
    # Even when their headers differ slightly, this produces consistent grids.
    tx = sitk.Transform(3, sitk.sitkIdentity)

    default_value = 0
    out = sitk.Resample(
        moving,
        ref,
        tx,
        interpolator,
        default_value,
        moving.GetPixelID(),
    )
    return out


def maybe_backup(path: str, backup_dir: str):
    if os.path.exists(path):
        ensure_dir(backup_dir)
        base = os.path.basename(path)
        dst = os.path.join(backup_dir, base)
        if not os.path.exists(dst):
            os.rename(path, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_raw", required=True, help="Path to nnUNet_raw/Dataset422_TDSI2025 (dataset root)")
    ap.add_argument("--ref_mod", type=int, default=2, help="Reference modality index (default 2 = FLAIR)")
    ap.add_argument("--mods", default="0,1,2", help="Comma-separated modality indices present (default '0,1,2')")
    ap.add_argument("--fix_labels", action="store_true", help="Also resample labelsTs to reference grid (nearest neighbor)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite imagesTs/labelsTs in place (creates backups/)")
    ap.add_argument("--tol", type=float, default=1e-6, help="Tolerance for grid equality checks")
    args = ap.parse_args()

    imagesTs = os.path.join(args.dataset_raw, "imagesTs")
    labelsTs = os.path.join(args.dataset_raw, "labelsTs")

    if not os.path.isdir(imagesTs):
        raise RuntimeError(f"imagesTs not found: {imagesTs}")

    mods = [int(x.strip()) for x in args.mods.split(",") if x.strip() != ""]
    if args.ref_mod not in mods:
        raise ValueError(f"ref_mod={args.ref_mod} not in mods={mods}")

    case_ids = list_case_ids(imagesTs, args.ref_mod)
    if len(case_ids) == 0:
        raise RuntimeError(f"No cases found with ref modality _{args.ref_mod:04d} in {imagesTs}")

    if args.inplace:
        out_imagesTs = imagesTs
        out_labelsTs = labelsTs
        backup_dir = os.path.join(args.dataset_raw, "backups_before_affine_fix")
        ensure_dir(backup_dir)
    else:
        out_imagesTs = os.path.join(args.dataset_raw, "imagesTs_fixed")
        out_labelsTs = os.path.join(args.dataset_raw, "labelsTs_fixed")
        ensure_dir(out_imagesTs)
        if args.fix_labels:
            ensure_dir(out_labelsTs)

    print(f"[INFO] dataset_raw : {args.dataset_raw}")
    print(f"[INFO] imagesTs     : {imagesTs}")
    print(f"[INFO] ref modality : {args.ref_mod:04d}")
    print(f"[INFO] modalities   : {mods}")
    print(f"[INFO] cases found  : {len(case_ids)}")
    print(f"[INFO] mode         : {'INPLACE' if args.inplace else 'WRITE_FIXED_FOLDERS'}")
    if args.fix_labels:
        print(f"[INFO] labelsTs     : {labelsTs} (will be resampled)")
    else:
        print("[INFO] labelsTs     : (not processed)")

    n_done = 0
    n_resampled_imgs = 0
    n_resampled_lbls = 0
    missing = []

    for cid in case_ids:
        ref_path = get_mod_path(imagesTs, cid, args.ref_mod)
        if not ref_path:
            missing.append((cid, "missing reference modality"))
            continue

        ref_img = sitk_read(ref_path)

        # Process modalities
        for m in mods:
            in_path = get_mod_path(imagesTs, cid, m)
            if not in_path:
                missing.append((cid, f"missing modality {m:04d}"))
                continue

            out_path = os.path.join(out_imagesTs, os.path.basename(in_path)) if not args.inplace else in_path

            mov = sitk_read(in_path)

            if same_grid(mov, ref_img, tol=args.tol):
                # Still ensure saved in fixed folder if not inplace
                if not args.inplace:
                    # Copy by writing (keeps exact header)
                    sitk.WriteImage(mov, out_path, True)
                continue

            # Need resample
            res = resample_to_ref(mov, ref_img, is_label=False)

            if args.inplace:
                maybe_backup(in_path, backup_dir)

            sitk.WriteImage(res, out_path, True)
            n_resampled_imgs += 1

        # Process label if asked
        if args.fix_labels:
            if not os.path.isdir(labelsTs):
                missing.append((cid, "labelsTs folder not found"))
            else:
                lbl_in = get_label_path(labelsTs, cid)
                if not lbl_in:
                    missing.append((cid, "missing label"))
                else:
                    lbl_out = os.path.join(out_labelsTs, os.path.basename(lbl_in)) if not args.inplace else lbl_in
                    lbl = sitk_read(lbl_in)

                    if not same_grid(lbl, ref_img, tol=args.tol):
                        lbl_res = resample_to_ref(lbl, ref_img, is_label=True)

                        if args.inplace:
                            maybe_backup(lbl_in, backup_dir)

                        sitk.WriteImage(lbl_res, lbl_out, True)
                        n_resampled_lbls += 1
                    else:
                        if not args.inplace:
                            sitk.WriteImage(lbl, lbl_out, True)

        n_done += 1
        if n_done % 10 == 0 or n_done == len(case_ids):
            print(f"[INFO] processed {n_done}/{len(case_ids)}")

    print("\n[DONE]")
    print(f"  cases processed           : {n_done}")
    print(f"  modalities resampled files: {n_resampled_imgs}")
    if args.fix_labels:
        print(f"  labels resampled files    : {n_resampled_lbls}")

    if missing:
        print(f"\n[WARN] missing/issues for {len(missing)} entries (first 20):")
        for x in missing[:20]:
            print(" ", x)

    if not args.inplace:
        print(f"\n[OUTPUT]")
        print(f"  imagesTs_fixed: {out_imagesTs}")
        if args.fix_labels:
            print(f"  labelsTs_fixed: {out_labelsTs}")
        print("\nNext step:")
        print("  - Run inference/benchmark using *_fixed folders")
    else:
        print("\n[INPLACE NOTE]")
        print(f"  Backups (renamed originals) in: {backup_dir}")
        print("  Now you can rerun inference directly on imagesTs/labelsTs")


if __name__ == "__main__":
    main()

