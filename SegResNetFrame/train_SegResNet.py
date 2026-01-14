#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train SegResNet (MONAI) for 3D binary segmentation from an nnU-Net raw dataset

This script trains ONE fold:
- If --use_splits is provided (splits_final.json), uses that fold's train/val IDs (nnU-Net style CV).
- Else uses a deterministic 80/20 random split.

Binary setup (recommended for MS lesions):
- SegResNet out_channels=1 (single logit)
- DiceCELoss(sigmoid=True)
- Validation uses sigmoid + threshold 0.5 for Dice

Run examples:
  conda activate monai-dev
  python train_SegResNet.py \
    --nnunet_raw /path/to/nnUNet_raw/Dataset421_TDSI2025 \
    --out /path/to/results/segresnet_fold0 \
    --modalities 1 \
    --use_splits /path/to/nnUNet_preprocessed/Dataset421_TDSI2025/splits_final.json \
    --fold 0 \
    --epochs 300 \
    --patch 96,96,96 \
    --batch_size 1 \ e 
    --resume
    --lr 1e-4 >> lr plays a huge role inoverflow. if you see loss = nan, it means lr might be too low. Try 1e-4 for dataset 421 and 3e-3 for dataset 422
"""

import os
import json
import argparse
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandAffined,
    DivisiblePadd
)
from monai.data import CacheDataset
from monai.data.utils import list_data_collate
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


def list_case_ids(images_dir: str) -> list[str]:
    """
    Get unique case ids from files like CASE_0000.nii.gz.
    Assumes modality 0000 exists for all cases.
    """
    files = sorted(glob(os.path.join(images_dir, "*_0000.nii*")))
    case_ids: list[str] = []
    for f in files:
        base = os.path.basename(f)
        # Split at '_0000.nii' or '_0000.nii.gz'
        case_id = base.split("_0000.nii")[0]
        case_ids.append(case_id)
    return case_ids


def build_items(case_ids: list[str], images_dir: str, labels_dir: str, modalities: list[int]) -> list[dict]:
    """
    Build list of items:
      {"image":[mod_paths...], "label": label_path, "case_id": str}

    MONAI LoadImaged can load a list of image paths and stack into channels.
    """
    items: list[dict] = []
    for cid in case_ids:
        img_paths: list[str] = []
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

        lbl_gz = os.path.join(labels_dir, f"{cid}.nii.gz")
        lbl_ni = os.path.join(labels_dir, f"{cid}.nii")
        if os.path.exists(lbl_gz):
            label_path = lbl_gz
        elif os.path.exists(lbl_ni):
            label_path = lbl_ni
        else:
            continue

        items.append({"image": img_paths, "label": label_path, "case_id": cid})

    return items


def load_nnunet_split(splits_json_path: str, fold: int) -> tuple[list[str], list[str]]:
    """
    Load nnU-Net splits_final.json and return (train_ids, val_ids) for fold.
    """
    with open(splits_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    if fold < 0 or fold >= len(splits):
        raise RuntimeError(f"Fold {fold} out of range in {splits_json_path} (n_folds={len(splits)})")

    train_ids = splits[fold]["train"]
    val_ids = splits[fold]["val"]
    return train_ids, val_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunet_raw", required=True, help="Path to nnUNet_raw/DatasetXXX_NAME")
    parser.add_argument("--out", required=True, help="Output directory for checkpoints/logs")
    parser.add_argument(
        "--modalities",
        default="1",
        help="Comma-separated modality indices (e.g. '1' for FLAIR, '0,1' for T1+FLAIR)",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patch", default="96,96,96", help="Patch size D,H,W e.g. '96,96,96'")
    parser.add_argument("--samples_per_image", type=int, default=4, help="Samples per image for RandCropByPosNegLabeld")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cache", type=float, default=0.2, help="CacheDataset cache_rate")
    parser.add_argument("--use_splits", default="", help="Path to splits_final.json (optional)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index if using splits_final.json")
    parser.add_argument("--resume", action="store_true", help="Resume from latest.pt if present")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    imagesTr = os.path.join(args.nnunet_raw, "imagesTr")
    labelsTr = os.path.join(args.nnunet_raw, "labelsTr")
    if not (os.path.isdir(imagesTr) and os.path.isdir(labelsTr)):
        raise RuntimeError("imagesTr/labelsTr not found. Check --nnunet_raw path.")

    modalities = [int(x.strip()) for x in args.modalities.split(",") if x.strip() != ""]
    in_channels = len(modalities)
    patch = tuple(int(x) for x in args.patch.split(","))


    # Split train/val case IDs

    all_case_ids = list_case_ids(imagesTr)
    if len(all_case_ids) == 0:
        raise RuntimeError(f"No cases found in {imagesTr}. Do you have *_0000.nii.gz files?")

    if args.use_splits:
        tr_ids, va_ids = load_nnunet_split(args.use_splits, args.fold)
        # Keep only those actually present in imagesTr
        tr_ids = [cid for cid in tr_ids if cid in all_case_ids]
        va_ids = [cid for cid in va_ids if cid in all_case_ids]
    else:
        rng = np.random.default_rng(0)
        ids = np.array(all_case_ids)
        rng.shuffle(ids)
        n_val = max(1, int(0.2 * len(ids)))
        va_ids = ids[:n_val].tolist()
        tr_ids = ids[n_val:].tolist()

    train_files = build_items(tr_ids, imagesTr, labelsTr, modalities)
    val_files = build_items(va_ids, imagesTr, labelsTr, modalities)

    if len(train_files) == 0 or len(val_files) == 0:
        raise RuntimeError(f"Empty split: train={len(train_files)} val={len(val_files)}. Check modalities/paths.")

    print(f"device={device}")
    print(f"in_channels={in_channels} modalities={modalities}")
    print(f"train_cases={len(train_files)} val_cases={len(val_files)} patch={patch}")

 
    # Transforms

    train_tf = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.int64),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            DivisiblePadd(keys=["image", "label"], k=16),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch,
                pos=1,
                neg=1,
                num_samples=args.samples_per_image,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["image", "label"],
                prob=0.2,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
        ]
    )

    val_tf = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.int64),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            DivisiblePadd(keys=["image", "label"], k=16),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]
    )

    # CacheDataset is fine; for huge datasets reduce cache_rate
    train_ds = CacheDataset(train_files, train_tf, cache_rate=args.cache, num_workers=args.num_workers)
    val_ds = CacheDataset(val_files, val_tf, cache_rate=min(args.cache, 0.1), num_workers=args.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate,  # IMPORTANT with RandCropByPosNegLabeld num_samples>1
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate,
    )

   
    # Model / Loss / Optim

    model = SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,  # binary: single logit
        init_filters=32,
        dropout_prob=0.0,
    ).to(device)

    loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True)  # binary
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # AMP (updated API)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    dice_metric = DiceMetric(include_background=True, reduction="mean")

    latest_ckpt = os.path.join(args.out, "latest.pt")
    best_ckpt = os.path.join(args.out, "best.pt")

    best_dice = -1.0
    start_epoch = 0

    if args.resume and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scaler.load_state_dict(ckpt["scaler"])
        best_dice = float(ckpt.get("best_dice", -1.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f" Resumed from {latest_ckpt} at epoch {start_epoch}, best_dice={best_dice:.4f}")

  
    # Training loop
  
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            #print("GT voxels: ", batch["label"].sum().item())
            #inter = (batch["image"][:,-1]>0) & (batch["label"]>0)
            #print("intersection voxels" , inter.sum().item())

            # Ensure label is (B,1,D,H,W)
            if y.ndim == 4:
                y = y.unsqueeze(1)
            y = (y > 0).to(torch.float32)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            #gradient  clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 12.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            n_steps += 1

        epoch_loss /= max(1, n_steps)

   
        # Validation
  
        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                """
                logits = model(x)
                print("logits.shape = ", tuple(logits.shape))
                print("logits min/ max = " , float(logits.min()),float(logits.max()))
                print("label unique = ", torch.unique(y)[:10])
                gt_pos = int((y>0).sum().item())
                print("gt_pos_vox =", gt_pos)
                
                if logits.shape[1] == 1:
                    prob = torch.sigmoid(logits)
                    pred = (prob > 0.5).to(y.dtype)
                else:
                    prob = torch.softmax(logits, dim = 1)
                    torch.argmax(prob, dim=1, keepdim=True).to(y.dtype)
                pred_pos = int((pred > 0).sum().item())
                inter = int(((pred>0) & (y>0)).sum().item())
                print("pred_pos_vox = ", pred_pos, "| intersection = ", inter)
                """

                if y.ndim == 4:
                    y = y.unsqueeze(1)
                y_bin = (y > 0).to(torch.uint8)

                logits = sliding_window_inference(x, roi_size=patch, sw_batch_size=1, predictor=model)
                pred_bin = (torch.sigmoid(logits) > 0.5).to(torch.uint8)

                # Debug once
                if epoch == 0:
                    gt_pos = int(y_bin.sum().item())
                    pr_pos = int(pred_bin.sum().item())
                    print(
                        f" val GT_pos_vox={gt_pos} | PRED_pos_vox={pr_pos} | "
                        f"logits_min={float(logits.min().item()):.3f} logits_max={float(logits.max().item()):.3f}"
                    )

                dice_metric(y_pred=pred_bin, y=y_bin)

            val_dice = float(dice_metric.aggregate().item())

        print(f"Epoch {epoch:04d}/{args.epochs} | train_loss={epoch_loss:.4f} | val_dice={val_dice:.4f}")

        # Save latest
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_dice": best_dice,
                "modalities": modalities,
                "patch": patch,
            },
            latest_ckpt,
        )

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_dice": best_dice,
                    "modalities": modalities,
                    "patch": patch,
                },
                best_ckpt,
            )
            print(f"New best val_dice={best_dice:.4f} -> saved {best_ckpt}")

    print(f"Training finished. Best val dice = {best_dice:.4f}")
    print(f"  best  : {best_ckpt}")
    print(f"  latest: {latest_ckpt}")


if __name__ == "__main__":
    main()

