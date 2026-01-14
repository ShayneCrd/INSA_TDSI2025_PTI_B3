#!/usr/bin/env python3
"""
Fix label origin/affine mismatch for Dataset424_TDSI2025.

For each case:
- uses FLAIR image (_0003) as reference
- resamples the label with nearest-neighbor interpolation
- overwrites the label in labelsTr / labelsTs

This resolves nnUNet verify_dataset_integrity errors.
"""

import os
from pathlib import Path
import SimpleITK as sitk


NNUNET_ROOT = Path(__file__).resolve().parent
DATASET_DIR = NNUNET_ROOT / "nnUNet_raw/Dataset424_TDSI2025"
IMAGES_TR = DATASET_DIR / "imagesTr"
LABELS_TR = DATASET_DIR / "labelsTr"
IMAGES_TS = DATASET_DIR / "imagesTs"
LABELS_TS = DATASET_DIR / "labelsTs"

# FLAIR channel index
FLAIR_CHANNEL = 3  # OPENMS2_xxxx_0003.nii.gz



def resample_label_to_reference(label_path: Path, ref_image_path: Path) -> None:
    """
    Resample label to reference image grid using nearest neighbor interpolation.
    Overwrites label_path.
    """
    label_img = sitk.ReadImage(str(label_path), sitk.sitkUInt8)
    ref_img = sitk.ReadImage(str(ref_image_path), sitk.sitkFloat32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    fixed_label = resampler.Execute(label_img)
    sitk.WriteImage(fixed_label, str(label_path))


def process_split(images_dir: Path, labels_dir: Path, split_name: str) -> None:
    print(f"\n=== Processing {split_name} ===")

    if not labels_dir.exists():
        print(f"{labels_dir} does not exist, skipping.")
        return

    labels = sorted(labels_dir.glob("*.nii*"))
    if not labels:
        print(f" No labels found in {labels_dir}, skipping.")
        return

    fixed = 0
    missing_ref = 0

    for lbl in labels:
        case_id = lbl.stem.replace(".nii", "")
        flair_img = images_dir / f"{case_id}_{FLAIR_CHANNEL:04d}.nii.gz"

        if not flair_img.exists():
            print(f"[WARN] Missing FLAIR for {case_id}, skipping.")
            missing_ref += 1
            continue

        resample_label_to_reference(lbl, flair_img)
        fixed += 1

    print(f"[OK] {fixed} labels fixed in {split_name}")
    if missing_ref > 0:
        print(f"{missing_ref} cases skipped due to missing FLAIR")




def main():
    print("==============================================")
    print(" Fixing label origin/affine for Dataset424 ")
    print(" Reference modality: FLAIR (_0003)")
    print("==============================================")

    process_split(IMAGES_TR, LABELS_TR, "labelsTr")
    process_split(IMAGES_TS, LABELS_TS, "labelsTs")

    print("\nDone.")



if __name__ == "__main__":
    main()

