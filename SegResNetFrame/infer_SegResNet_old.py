#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_segresnet.py

Run inference with a trained MONAI SegResNet (binary, out_channels=1) on an nnU-Net raw dataset.

Expected nnU-Net raw structure:
  DatasetXXX_NAME/
    ├── imagesTs/   CASE_0000.nii.gz, CASE_0001.nii.gz, ...
    └── (optional labelsTs/)

This script:
- Loads a checkpoint (best.pt or latest.pt produced by the training script)
- Builds a list of test cases from imagesTs using modality indices
- Applies deterministic preprocessing (same as validation):
    LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd(RAS), NormalizeIntensityd
- Uses sliding_window_inference for full-volume prediction
- Saves:
    - probability map (float32) as *_prob.nii.gz (optional)
    - binary mask (uint8) as *_mask.nii.gz

Usage:
  conda activate monai-dev
  python infer_segresnet.py \
    --nnunet_raw /path/to/nnUNet_raw/Dataset421_TDSI2025 \
    --ckpt /path/to/results/segresnet_fold0/best.pt \
    --out /path/to/preds/segresnet_fold0_imagesTs \
    --modalities 1 \
    --patch 96,96,96 \
    --save_prob

Notes:
- --modalities must match training (e.g. 1 for FLAIR, or 0,1 for T1+FLAIR)
- Patch size controls ROI for sliding window. Bigger can be faster but uses more VRAM.
"""

import os
import argparse
from glob import glob

import numpy as np
import torch
import nibabel as nib

from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    NormalizeIntensityd,
)
from monai.data import Dataset
from monai.data.utils import list_data_collate
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet


def list_case_ids(images_dir: str) -> list[str]:
    """
    Get unique case ids from files like CASE_0000.nii.gz.
    Assumes modality 0000 exists for all cases.
    If you trained with modalities not including 0000, we still use 0000 for listing cases;
    if your imagesTs doesn't have 0000, set --list_modality to one you have.
    """
    files = sorted(glob(os.path.join(images_dir, "*_0000.nii*")))
    case_ids = []
    for f in files:
        base = os.path.basename(f)
        cid = base.split("_0000.nii")[0]
        case_ids.append(cid)
    return case_ids


def list_case_ids_by_modality(images_dir: str, modality_idx: int) -> list[str]:
    files = sorted(glob(os.path.join(images_dir, f"*_{modality_idx:04d}.nii*")))
    case_ids = []
    for f in files:
        base = os.path.basename(f)
        cid = base.split(f"_{modality_idx:04d}.nii")[0]
        case_ids.append(cid)
    return case_ids


def build_items(case_ids: list[str], images_dir: str, modalities: list[int]) -> list[dict]:
    """
    items: {"image": [mod_paths...], "case_id": str, "ref_nii": path_to_first_modality}
    """
    items = []
    for cid in case_ids:
        img_paths = []
        ok = True
        for m in modalities:
            p_gz = os.path.join(images_dir, f"{cid}_{m:04d}.nii.gz")
            p_ni = os.path.join(images_dir, f"{cid}_{m:04d}.nii")
            if os.path.exists(p_gz):
                img_paths.append(p_gz)
            elif os.path.exists(p_ni):
                img_paths.append(p_ni)
            else:
                ok = False
                break
        if not ok:
            continue

        # reference image for affine/header (use first modality)
        ref_nii = img_paths[0]
        items.append({"image": img_paths, "case_id": cid, "ref_nii": ref_nii})
    return items


def save_nifti_like(ref_nii_path: str, data: np.ndarray, out_path: str, dtype):
    """
    Save `data` as NIfTI using affine/header from ref_nii_path.
    data should be (D,H,W). Will cast to dtype.
    """
    ref_img = nib.load(ref_nii_path)
    out = nib.Nifti1Image(data.astype(dtype), ref_img.affine, ref_img.header)
    nib.save(out, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnunet_raw", required=True, help="Path to nnUNet_raw/DatasetXXX_NAME")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (best.pt or latest.pt)")
    ap.add_argument("--out", required=True, help="Output folder for predictions")
    ap.add_argument("--modalities", default="1", help="Comma-separated modality indices used at training (e.g. '1' or '0,1')")
    ap.add_argument("--patch", default="96,96,96", help="ROI size for sliding window inference D,H,W")
    ap.add_argument("--sw_batch_size", type=int, default=1, help="Sliding window batch size (increase if VRAM allows)")
    ap.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap [0..1]")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask on prob map")
    ap.add_argument("--save_prob", action="store_true", help="Save probability maps (*_prob.nii.gz)")
    ap.add_argument("--list_modality", type=int, default=0, help="Modality index used to list test cases (default 0)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    imagesTs = os.path.join(args.nnunet_raw, "imagesTs")
    if not os.path.isdir(imagesTs):
        raise RuntimeError(f"imagesTs not found in {args.nnunet_raw}")

    modalities = [int(x.strip()) for x in args.modalities.split(",") if x.strip() != ""]
    in_channels = len(modalities)
    patch = tuple(int(x) for x in args.patch.split(","))

    # Build list of test cases
    case_ids = list_case_ids_by_modality(imagesTs, args.list_modality)
    if len(case_ids) == 0:
        raise RuntimeError(
            f"No test cases found using list modality {args.list_modality} in {imagesTs}. "
            f"Check your files exist like *_000{args.list_modality}.nii.gz"
        )

    test_files = build_items(case_ids, imagesTs, modalities)
    if len(test_files) == 0:
        raise RuntimeError(
            f"No valid test items built. Probably missing one of modalities {modalities} in imagesTs."
        )

    print(f"[INFO] device={device}")
    print(f"[INFO] modalities={modalities} in_channels={in_channels}")
    print(f"[INFO] test_cases={len(test_files)} roi_size={patch} overlap={args.overlap}")

    # Deterministic inference transforms
    infer_tf = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),  # stacks modalities -> (C,D,H,W)
            EnsureTyped(keys=["image"], dtype=torch.float32),
            #Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]
    )

    ds = Dataset(test_files, transform=infer_tf)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate,
    )

    # Build model (binary, out_channels=1)
    model = SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        init_filters=32,
        dropout_prob=0.0,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        # allow direct state_dict
        model.load_state_dict(ckpt)

    model.eval()

    # Inference loop
    n_saved = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)  # (1,C,D,H,W)
            case_id = batch["case_id"][0]  # string
            ref_nii = batch["ref_nii"][0]  # path

            logits = sliding_window_inference(
                inputs=x,
                roi_size=patch,
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=args.overlap,
            )

            prob = torch.sigmoid(logits)  # (1,1,D,H,W)
            mask = (prob > args.threshold).to(torch.uint8)

            prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
            mask_np = mask[0, 0].detach().cpu().numpy().astype(np.uint8)

            # Save outputs
            out_mask = os.path.join(args.out, f"{case_id}.nii.gz")
            save_nifti_like(ref_nii, mask_np, out_mask, np.uint8)

            if args.save_prob:
                out_prob = os.path.join(args.out, f"{case_id}_prob.nii.gz")
                save_nifti_like(ref_nii, prob_np, out_prob, np.float32)

            n_saved += 1
            if n_saved % 10 == 0 or n_saved == len(test_files):
                print(f"[INFO] Saved {n_saved}/{len(test_files)}")

    print(f"[DONE] Saved predictions to: {args.out}")
    print(f"       Total cases: {n_saved}")


if __name__ == "__main__":
    main()

