#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_segresnet.py

Inference for MONAI SegResNet on nnU-Net raw dataset (imagesTs) with:
- single checkpoint mode (default)
- cross-val ensemble mode (5 folds) with ONLINE averaging (no per-fold prediction folders)

Ensemble logic (recommended):
- run each fold model -> logits_k
- mean_logits = average(logits_k)
- prob = sigmoid(mean_logits)
- mask = prob > threshold
Saves only:
- final mask: CASEID.nii.gz (uint8)
- optional final probmap: CASEID_prob.nii.gz (float32)

Usage (single model):
  python infer_segresnet.py \
    --nnunet_raw /path/to/nnUNet_raw/Dataset421_TDSI2025 \
    --ckpt /path/to/fold0/best.pt \
    --out /path/to/preds_single \
    --modalities 1 \
    --patch 96,96,96 \
    --save_prob

Usage (CV ensemble, online):
  python infer_segresnet.py \
    --nnunet_raw /path/to/nnUNet_raw/Dataset421_TDSI2025 \
    --cv_dir /path/to/SegResNet_results/Dataset421_results \
    --out /path/to/preds_cv_ensemble \
    --modalities 0,1 \
    --patch 96,96,96 \
    --save_prob \
    --use_cv

Expected cv_dir structure (flexible):
- If you provide --cv_dir, script will search for checkpoints in common patterns:
    {cv_dir}/fold_0/best.pt
    {cv_dir}/fold_0/latest.pt
    {cv_dir}/fold0/best.pt
    {cv_dir}/fold0/latest.pt
    {cv_dir}/best_fold0.pt
    {cv_dir}/latest_fold0.pt
and similarly for folds 1..4.
You can override with --ckpts "path0,path1,path2,path3,path4"
"""

import os
import re
import argparse
from glob import glob
from typing import List, Dict, Optional
from torch.cuda.amp import autocast

import numpy as np
import torch
import nibabel as nib

from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
)
from monai.data import Dataset
from monai.data.utils import list_data_collate
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet


_CASE_RE = re.compile(r"^(?P<cid>.+)_(?P<ch>\d{4})\.nii(\.gz)?$")


def list_case_ids_by_modality(images_dir: str, modality_idx: int) -> List[str]:
    files = sorted(glob(os.path.join(images_dir, f"*_{modality_idx:04d}.nii*")))
    case_ids = []
    for f in files:
        base = os.path.basename(f)
        cid = base.split(f"_{modality_idx:04d}.nii")[0]
        case_ids.append(cid)
    return case_ids


def build_items(case_ids: List[str], images_dir: str, modalities: List[int]) -> List[Dict]:
    """
    items: {"image": [mod_paths...], "case_id": str, "ref_nii": path_to_first_modality}
    MONAI LoadImaged can load a list of image paths and stack into channels.
    """
    items: List[Dict] = []
    for cid in case_ids:
        img_paths: List[str] = []
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


# Checkpoint discovery

def _candidate_ckpt_paths(cv_dir: str, fold: int) -> List[str]:
    """
    Return candidate checkpoint paths in common layouts.
    """
    cand = [
        os.path.join(cv_dir, f"fold_{fold}", "best.pt"),
        os.path.join(cv_dir, f"fold_{fold}", "latest.pt"),
        os.path.join(cv_dir, f"fold{fold}", "best.pt"),
        os.path.join(cv_dir, f"fold{fold}", "latest.pt"),
        os.path.join(cv_dir, f"best_fold{fold}.pt"),
        os.path.join(cv_dir, f"latest_fold{fold}.pt"),
        os.path.join(cv_dir, f"fold_{fold}", "checkpoint_best.pt"),
        os.path.join(cv_dir, f"fold_{fold}", "checkpoint_latest.pt"),
    ]
    return cand


def discover_cv_ckpts(cv_dir: str, folds: List[int], prefer: str = "best") -> List[str]:
    """
    Discover ckpt per fold from cv_dir.
    prefer: "best" or "latest"
    """
    ckpts: List[str] = []
    missing: List[int] = []

    for f in folds:
        cands = _candidate_ckpt_paths(cv_dir, f)
        # reorder by preference
        if prefer == "latest":
            cands = sorted(cands, key=lambda p: ("latest" not in os.path.basename(p).lower(), p))
        else:
            cands = sorted(cands, key=lambda p: ("best" not in os.path.basename(p).lower(), p))

        found = None
        for p in cands:
            if os.path.exists(p):
                found = p
                break

        if found is None:
            missing.append(f)
        else:
            ckpts.append(found)

    if missing:
        raise RuntimeError(
            f"Could not find checkpoints for folds {missing} in cv_dir={cv_dir}. "
            f"Either fix folder layout or pass --ckpts with explicit paths."
        )

    return ckpts



# Model loading

def build_model(in_channels: int, device: torch.device) -> SegResNet:
    # binary segmentation: out_channels=1
    return SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        init_filters=32,
        dropout_prob=0.0,
    ).to(device)


def load_ckpt_into_model(model: SegResNet, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)


def infer_logits(
    model: SegResNet,
    x: torch.Tensor,
    patch: tuple,
    sw_batch_size: int,
    overlap: float,
) -> torch.Tensor:
    # returns logits (1,1,D,H,W)
    logits = sliding_window_inference(
        inputs=x,
        roi_size=patch,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )
    return logits



# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnunet_raw", required=True, help="Path to nnUNet_raw/DatasetXXX_NAME")
    ap.add_argument("--out", required=True, help="Output folder for final predictions")

    # single-ckpt
    ap.add_argument("--ckpt", default=None, help="Single checkpoint path (best.pt / latest.pt)")

    # CV ensemble
    ap.add_argument("--use_cv", action="store_true", help="Enable CV ensemble inference (multiple folds)")
    ap.add_argument("--cv_dir", default=None, help="Folder containing per-fold checkpoints (see docstring)")
    ap.add_argument("--folds", default="0,1,2,3,4", help="Comma-separated folds for ensemble, e.g. '0,1,2,3,4'")
    ap.add_argument("--ckpts", default=None, help="Explicit comma-separated ckpt paths for folds")
    ap.add_argument("--prefer", default="best", choices=["best", "latest"], help="When discovering ckpts from cv_dir")

    # data / inference params
    ap.add_argument("--modalities", default="1", help="Comma-separated modality indices used at training (e.g. '1' or '0,1')")
    ap.add_argument("--patch", default="96,96,96", help="ROI size for sliding window inference D,H,W")
    ap.add_argument("--sw_batch_size", type=int, default=1, help="Sliding window batch size (increase if VRAM allows)")
    ap.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap [0..1]")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask on prob map")
    ap.add_argument("--save_prob", action="store_true", help="Save averaged probability maps (*_prob.nii.gz)")
    ap.add_argument("--list_modality", type=int, default=0, help="Modality index used to list test cases (default 0)")

    # performance
    ap.add_argument("--preload_models", action="store_true",
                    help="Preload all fold models into memory (faster, uses more VRAM/RAM). "
                         "If false, loads weights fold-by-fold into a single model (slower, minimal VRAM).")
    ap.add_argument("--amp", action="store_true", help="Use autocast for faster inference on GPU (recommended).")

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
            f"Check files exist like *_ {args.list_modality:04d}.nii.gz"
        )

    test_files = build_items(case_ids, imagesTs, modalities)
    if len(test_files) == 0:
        raise RuntimeError(f"No valid test items built. Missing one of modalities {modalities} in imagesTs?")

    print(f"[INFO] device={device}")
    print(f"[INFO] modalities={modalities} in_channels={in_channels}")
    print(f"[INFO] test_cases={len(test_files)} roi_size={patch} overlap={args.overlap}")

    # Deterministic inference transforms (match val-style)
    infer_tf = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),  # stacks modalities -> (C,D,H,W)
            EnsureTyped(keys=["image"], dtype=torch.float32),
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

    # Decide checkpoints
    if args.use_cv:
        folds = [int(x.strip()) for x in args.folds.split(",") if x.strip() != ""]
        if args.ckpts is not None:
            ckpt_paths = [p.strip() for p in args.ckpts.split(",") if p.strip() != ""]
            if len(ckpt_paths) != len(folds):
                raise RuntimeError(f"--ckpts has {len(ckpt_paths)} paths but --folds has {len(folds)} folds.")
        else:
            if args.cv_dir is None:
                raise RuntimeError("--use_cv requires either --ckpts or --cv_dir.")
            ckpt_paths = discover_cv_ckpts(args.cv_dir, folds, prefer=args.prefer)

        print("[INFO] CV ensemble enabled.")
        for f, p in zip(folds, ckpt_paths):
            print(f"       fold {f}: {p}")
    else:
        if args.ckpt is None:
            raise RuntimeError("Provide --ckpt for single-model inference or use --use_cv with --cv_dir/--ckpts.")
        ckpt_paths = [args.ckpt]
        print(f"[INFO] Single-model inference: ckpt={args.ckpt}")

    # Build model(s)
    if args.use_cv and args.preload_models:
        models: List[SegResNet] = []
        for p in ckpt_paths:
            m = build_model(in_channels, device)
            load_ckpt_into_model(m, p, device)
            m.eval()
            models.append(m)
    else:
        # single model reused (load weights fold-by-fold if CV)
        model = build_model(in_channels, device)
        # if not CV, load once now
        if not args.use_cv:
            load_ckpt_into_model(model, ckpt_paths[0], device)
            model.eval()

    # Inference loop
    n_saved = 0
    autocast_ctx = torch.cuda.amp.autocast if (device.type == "cuda" and args.amp) else torch.cpu.amp.autocast

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)  # (1,C,D,H,W)
            case_id = batch["case_id"][0]
            ref_nii = batch["ref_nii"][0]

            if args.use_cv:
                # Online ensemble: accumulate mean logits, no per-fold disk outputs
                logits_sum: Optional[torch.Tensor] = None
                n_models = len(ckpt_paths)

                for idx, ckpt_path in enumerate(ckpt_paths):
                    if args.preload_models:
                        m = models[idx]
                    else:
                        # load fold weights into the single model
                        load_ckpt_into_model(model, ckpt_path, device)
                        model.eval()
                        m = model

                    with autocast(enabled=(device.type == "cuda")):
                        logits = infer_logits(m, x, patch, args.sw_batch_size, args.overlap)

                    if logits_sum is None:
                        logits_sum = logits.float()
                    else:
                        logits_sum = logits_sum + logits.float()

                mean_logits = logits_sum / float(n_models)
                prob = torch.sigmoid(mean_logits)  # (1,1,D,H,W)
            else:
                with autocast_ctx(device_type="cuda" if device.type == "cuda" else "cpu"):
                    logits = infer_logits(model, x, patch, args.sw_batch_size, args.overlap)
                prob = torch.sigmoid(logits)

            mask = (prob > args.threshold).to(torch.uint8)

            prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
            mask_np = mask[0, 0].detach().cpu().numpy().astype(np.uint8)

            # Save outputs (final only)
            out_mask = os.path.join(args.out, f"{case_id}.nii.gz")
            save_nifti_like(ref_nii, mask_np, out_mask, np.uint8)

            if args.save_prob:
                out_prob = os.path.join(args.out, f"{case_id}_prob.nii.gz")
                save_nifti_like(ref_nii, prob_np, out_prob, np.float32)

            n_saved += 1
            if n_saved % 10 == 0 or n_saved == len(test_files):
                print(f"[INFO] Saved {n_saved}/{len(test_files)}")

    print(f"Saved predictions to: {args.out}")
    print(f"       Total cases: {n_saved}")


if __name__ == "__main__":
    main()

