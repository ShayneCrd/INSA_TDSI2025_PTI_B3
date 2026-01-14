#!/usr/bin/env python3
import os
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import re



NNUNET_ROOT = Path(__file__).resolve().parent

NNUNET_RAW = NNUNET_ROOT / "nnUNet_raw"

# Dataset423 (longitudinal)
DS423 = NNUNET_RAW / "Dataset423_TDSI2025"
OPENMS_LONG_ROOT = NNUNET_ROOT / "open_ms_data-master/longitudinal/coregistered"

# Dataset424 (cross-sectional)
DS424 = NNUNET_RAW / "Dataset424_TDSI2025"
OPENMS2_CROSS_ROOT = NNUNET_RAW / "open_ms_data-master/cross_sectional/coregistered"
_CASE_RE = re.compile(r"^(?P<cid>.+)_(?P<ch>\d{4})\.nii(\.gz)?$")

# SETTINGS

OVERWRITE_MASKED = True     # overwrite images after masking
OVERWRITE_DIFF = False      # if diff channels exist, overwrite them
MASK_DTYPE = np.float32

# 423: channels 0..5 exist (6-channel input)
# We will create diff channels 6..8: (study2 - study1) for (T1,T2,FLAIR)
CH_T1_1, CH_T2_1, CH_FLAIR_1 = 0, 1, 2
CH_T1_2, CH_T2_2, CH_FLAIR_2 = 3, 4, 5
DIFF_CHANNELS = {
    6: (CH_T1_2, CH_T1_1),
    7: (CH_T2_2, CH_T2_1),
    8: (CH_FLAIR_2, CH_FLAIR_1),
}

# 424 channels 0..3 exist, just mask them
DS424_CHANNELS = [0, 1, 2, 3]



def load_nii(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))

def save_nii(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    out = nib.Nifti1Image(data.astype(MASK_DTYPE), ref_img.affine, ref_img.header)
    out.set_data_dtype(MASK_DTYPE)
    nib.save(out, str(out_path))

def find_patient_brainmask_longitudinal(case_id: str) -> Path:
    parts = case_id.split("_")
    if len(parts) < 2:
        raise RuntimeError(f"[423] Unexpected case_id format: '{case_id}'")

    # OPENMS_0002 -> pid=2 ; OPENMS_1002 -> pid=2 (si jamais)
    n = int(parts[1])
    pid = n % 1000

    pdir = OPENMS_LONG_ROOT / f"patient{pid:02d}"
    bm = pdir / "brainmask.nii.gz"
    if not bm.exists():
        bm = pdir / "brainmask.nii"
    if not bm.exists():
        raise FileNotFoundError(f"[423] brainmask not found for {case_id} in {pdir}")
    return bm


def find_patient_brainmask_cross(case_id: str) -> Path:
    """
    case_id like OPENMS2_0001 -> patient01/brainmask.nii.gz
    """
    pid = int(case_id.split("_")[1])
    pdir = OPENMS2_CROSS_ROOT / f"patient{pid:02d}"
    bm = pdir / "brainmask.nii.gz"
    if not bm.exists():
        bm = pdir / "brainmask.nii"
    if not bm.exists():
        raise FileNotFoundError(f"[424] brainmask not found for {case_id} in {pdir}")
    return bm

def get_case_ids(images_dir: Path) -> list[str]:
    """
    Return sorted unique case IDs from nnUNet-style files:
      CASEID_0000.nii.gz, CASEID_0001.nii.gz, ...
    Ex: OPENMS_0002_0000.nii.gz -> case_id = OPENMS_0002
    """
    case_ids = set()
    for f in images_dir.iterdir():
        if not f.is_file():
            continue
        m = _CASE_RE.match(f.name)
        if not m:
            continue
        case_ids.add(m.group("cid"))
    return sorted(case_ids)


def apply_brainmask_to_channel(img_path: Path, bm_path: Path, overwrite: bool) -> None:
    """
    image := image * (brainmask>0). Saves float32.
    """
    if (not overwrite) and img_path.exists():
        # we still need to apply if overwrite=False? here: no-op
        return

    img_nii = load_nii(img_path)
    bm_nii = load_nii(bm_path)

    img = img_nii.get_fdata(dtype=np.float32)
    bm = bm_nii.get_fdata(dtype=np.float32)

    if img.shape != bm.shape:
        raise RuntimeError(f"Shape mismatch: {img_path} {img.shape} vs brainmask {bm_path} {bm.shape}")

    bm_bin = (bm > 0).astype(np.float32)
    masked = img * bm_bin

    save_nii(masked, img_nii, img_path)

def compute_and_save_diff(out_path: Path, img_a_path: Path, img_b_path: Path, overwrite: bool) -> None:
    """
    out = A - B, float32.
    A and B must share same grid (shape/affine). Uses A header/affine.
    """
    if out_path.exists() and not overwrite:
        return

    a_nii = load_nii(img_a_path)
    b_nii = load_nii(img_b_path)

    a = a_nii.get_fdata(dtype=np.float32)
    b = b_nii.get_fdata(dtype=np.float32)

    if a.shape != b.shape:
        raise RuntimeError(f"Diff shape mismatch: {img_a_path} {a.shape} vs {img_b_path} {b.shape}")

    # NOTE: we assume aligned already (coregistered)
   

    diff = a - b
    save_nii(diff, a_nii, out_path)

def update_dataset_json_for_423(ds423_dir: Path) -> None:
    """
    Ensure dataset.json has channel_names 0..8.
    """
    js_path = ds423_dir / "dataset.json"
    if not js_path.exists():
        print(f"[WARN] {js_path} not found. Skipping dataset.json update.")
        return

    with open(js_path, "r") as f:
        js = json.load(f)

    ch = js.get("channel_names", {})
    # Set/update names
    ch.update({
        "0": "T1_t1",
        "1": "T2_t1",
        "2": "FLAIR_t1",
        "3": "T1_t2",
        "4": "T2_t2",
        "5": "FLAIR_t2",
        "6": "diff_T1",
        "7": "diff_T2",
        "8": "diff_FLAIR",
    })
    js["channel_names"] = ch

    with open(js_path, "w") as f:
        json.dump(js, f, indent=2)

    print(" Updated Dataset423 dataset.json channel_names to include diff channels 6..8")



def process_dataset423():
    print("\n==============================")
    print("Dataset423: brainmask + diff channels")
    print("==============================")

    for split in ["imagesTr", "imagesTs"]:
        images_dir = DS423 / split
        if not images_dir.exists():
            print(f"[WARN] Missing {images_dir}, skipping.")
            continue

        case_ids = get_case_ids(images_dir)
        print(f"[INFO] {split}: {len(case_ids)} cases")

        for cid in case_ids:
            bm = find_patient_brainmask_longitudinal(cid)

            # mask channels 0..5
            for ch in range(6):
                img = images_dir / f"{cid}_{ch:04d}.nii.gz"
                if not img.exists():
                    img = images_dir / f"{cid}_{ch:04d}.nii"
                if not img.exists():
                    raise FileNotFoundError(f"[423] Missing channel {ch} for {cid} in {images_dir}")
                apply_brainmask_to_channel(img, bm, overwrite=OVERWRITE_MASKED)

            # create diff channels 6..8
            for out_ch, (a_ch, b_ch) in DIFF_CHANNELS.items():
                a_path = images_dir / f"{cid}_{a_ch:04d}.nii.gz"
                if not a_path.exists():
                    a_path = images_dir / f"{cid}_{a_ch:04d}.nii"
                b_path = images_dir / f"{cid}_{b_ch:04d}.nii.gz"
                if not b_path.exists():
                    b_path = images_dir / f"{cid}_{b_ch:04d}.nii"

                out_path = images_dir / f"{cid}_{out_ch:04d}.nii.gz"
                compute_and_save_diff(out_path, a_path, b_path, overwrite=OVERWRITE_DIFF)

    update_dataset_json_for_423(DS423)
    print("Dataset423 processed.")


def process_dataset424():
    print("\n==============================")
    print("Dataset424: brainmask only")
    print("==============================")

    for split in ["imagesTr", "imagesTs"]:
        images_dir = DS424 / split
        if not images_dir.exists():
            print(f" Missing {images_dir}, skipping.")
            continue

        case_ids = get_case_ids(images_dir)
        print(f"{split}: {len(case_ids)} cases")

        for cid in case_ids:
            bm = find_patient_brainmask_cross(cid)

            for ch in DS424_CHANNELS:
                img = images_dir / f"{cid}_{ch:04d}.nii.gz"
                if not img.exists():
                    img = images_dir / f"{cid}_{ch:04d}.nii"
                if not img.exists():
                    raise FileNotFoundError(f"[424] Missing channel {ch} for {cid} in {images_dir}")

                apply_brainmask_to_channel(img, bm, overwrite=OVERWRITE_MASKED)

    print(" Dataset424 processed.")


def main():
    # basic guards
    """
    if not DS423.exists():
        print(f" {DS423} not found. Skipping Dataset423.")
    else:
        process_dataset423()
    """
    if not DS424.exists():
        print(f"{DS424} not found. Skipping Dataset424.")
    else:
        process_dataset424()

    print("\nNext steps:")
    print(" - For Dataset423: re-run nnUNetv2_plan_and_preprocess -d 423 --verify_dataset_integrity")
    print(" - For Dataset424: re-run nnUNetv2_plan_and_preprocess -d 424 --verify_dataset_integrity")


if __name__ == "__main__":
    main()

