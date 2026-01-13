#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import label


# ============================================================
# Metrics functions (copied from your spec)
# ============================================================

def load_masks(gt_path, pred_path):
    """
    Load ground-truth and predicted masks from NIfTI files.
We assule that the images have:
    - Same orientation
    - Same voxel space
    - Same shape
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT mask not found: {gt_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predicted mask not found: {pred_path}")

    gt_img = nib.load(gt_path)
    pred_img = nib.load(pred_path)

    gt = gt_img.get_fdata()
    pred = pred_img.get_fdata()

    # Handle possible 4D volumes
    if gt.ndim == 4:
        if gt.shape[-1] == 1:
            gt = gt[..., 0]
        else:
            raise ValueError(f"GT mask is 4D with unexpected shape: {gt.shape}")

    if pred.ndim == 4:
        if pred.shape[-1] == 1:
            pred = pred[..., 0]
        else:
            raise ValueError(f"Pred mask is 4D with unexpected shape: {pred.shape}")

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs Pred {pred.shape}")

    return gt, pred


def perform_dice(gt_mask, pred_mask, threshold=0.0):
    if gt_mask.shape != pred_mask.shape:
        #if there is a shape mismatch, we raise an error
        raise ValueError(f"Shape mismatch: GT {gt_mask.shape} vs Pred {pred_mask.shape}")

#we add a binarization to the ground truth mask and the prediction mask for safety, i.e to ensure the images are really binary
    gt_bin = gt_mask > threshold
    pred_bin = pred_mask > threshold

#counting non zero elements
    gt_sum = np.count_nonzero(gt_bin)
    pred_sum = np.count_nonzero(pred_bin)

#We handle limit cases where there's no ground truth lesion or no predicted lesion....
    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    if gt_sum == 0 and pred_sum > 0:
        return 0.0
    if gt_sum > 0 and pred_sum == 0:
        return 0.0
#We count all non-zero elements in the intersection between ground truth and prediction
    intersection = np.count_nonzero(gt_bin & pred_bin)
    dice = (2.0 * intersection) / (gt_sum + pred_sum) #bbased on the dice formula
    return float(dice)


def perform_volume_similarity(gt_mask, pred_mask, threshold=0.0):
    
    """
    This function counts the non-zero voxels for ground Ng truth and predicted masks Np
    Volume similarity is a comparison between Ng and Np.
    """
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(f"Shape mismatch: GT {gt_mask.shape} vs Pred {pred_mask.shape}")

    gt_bin = gt_mask > threshold
    pred_bin = pred_mask > threshold

    V_gt = int(np.count_nonzero(gt_bin))
    V_pred = int(np.count_nonzero(pred_bin))
    abs_diff = abs(V_gt - V_pred)

#Wee handle edge cases
    if V_gt == 0 and V_pred == 0:
        return 1.0, V_gt, V_pred, abs_diff
    if V_gt == 0 and V_pred > 0:
        return 0.0, V_gt, V_pred, abs_diff
    if V_gt > 0 and V_pred == 0:
        return 0.0, V_gt, V_pred, abs_diff

    vs = 1.0 - (abs_diff / (V_gt + V_pred))   #VS formula
    return float(vs), V_gt, V_pred, abs_diff


def perform_mAP_from_prob(
    gt_mask,
    pred_prob,
    iou_thresholds=(0.25, 0.5),
    connectivity=26,
    binarize_threshold=0.05,   # seuil fixe pour créer les instances (pas un sweep)
    min_cc_size=20,            # filtre bruit (20 vox est un bon début)
    score_mode="max",          # "max" | "mean" | "p95"
):
    """
    Lesion-wise mAP from a probability map using ranking AP.

    Pipeline:
      1) GT instances = CC(gt_mask>0)
      2) Pred instances = CC(pred_prob>binarize_threshold), filter min_cc_size
      3) Score each predicted instance (max/mean/p95 prob inside CC)
      4) Sort predictions by score desc
      5) For each IoU threshold tau:
          - we perform a greedy match in ranking order (each GT matched at most once)
          - build precision/recall curve cumulatively
          - AP = area under PR curve (VOC-style)

    Returns:
      {
        "mAP": float,
        "AP_per_iou": {tau: ap, ...},
        "metrics_per_iou": {tau: {...}, ...}
      }
    """
    if gt_mask.shape != pred_prob.shape:
        raise ValueError(f"Shape mismatch: GT {gt_mask.shape} vs Prob {pred_prob.shape}")

    # -Connectivty struct. We caracterize all the neighbour voxels for 1 voxel. Conectivity of 26 for a voxel is like
    #surrounding the voxel by a 3*3 cube
    if connectivity == 26:
        struct = np.ones((3, 3, 3), dtype=np.int8)
    elif connectivity == 6:
        struct = np.zeros((3, 3, 3), dtype=np.int8)
        struct[1, 1, 1] = 1
        struct[0, 1, 1] = struct[2, 1, 1] = 1
        struct[1, 0, 1] = struct[1, 2, 1] = 1
        struct[1, 1, 0] = struct[1, 1, 2] = 1
    else:
        raise ValueError("connectivity must be 6 or 26")

    # GT instances
    #safety binarization
    gt_bin = gt_mask > 0
    gt_lab, n_gt = label(gt_bin, structure=struct)
    if n_gt == 0:
        # no GT -> AP = 1 only if no predictions at all, else 0
        pred_any = np.any(pred_prob > binarize_threshold)
        ap = 0.0 if pred_any else 1.0
        return {
            "mAP": float(np.mean([ap for _ in iou_thresholds])),
            "AP_per_iou": {float(t): float(ap) for t in iou_thresholds},
            "metrics_per_iou": {float(t): {"n_gt": 0, "note": "n_gt=0 special case"} for t in iou_thresholds},
        }

    gt_sizes = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)

    # We binarize the predictions' probability maps 
    pred_bin = pred_prob > float(binarize_threshold)
    pred_lab, n_pred = label(pred_bin, structure=struct)
    pred_sizes = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)

    # filter small CC
    if n_pred > 0 and min_cc_size > 1:
        keep = pred_sizes >= int(min_cc_size)
        keep[0] = False
        pred_lab = keep[pred_lab]  # boolean mask on labels
        pred_lab, n_pred = label(pred_lab, structure=struct)  # relabel
        pred_sizes = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)

    # build list of predicted instances with scores
    pred_instances = []
    for p in range(1, n_pred + 1):
        if pred_sizes[p] == 0:
            continue
        vox = (pred_lab == p)
        vals = pred_prob[vox]

        if vals.size == 0:
            continue

        if score_mode == "max":
            score = float(vals.max())
        elif score_mode == "mean":
            score = float(vals.mean())
        elif score_mode == "p95":
            score = float(np.percentile(vals, 95))
        else:
            raise ValueError("score_mode must be 'max', 'mean', or 'p95'")

        pred_instances.append((p, score))

    # sort by confidence desc
    pred_instances.sort(key=lambda x: x[1], reverse=True)

    # helper: IoU between pred component p and gt component j
    def iou_pred_gt(p_label, j_label):
        pred_vox = (pred_lab == p_label)
        gt_overlap = gt_lab[pred_vox]
        if gt_overlap.size == 0:
            return 0.0, 0  # iou, intersection
        inter = int(np.count_nonzero(gt_overlap == j_label))
        if inter == 0:
            return 0.0, 0
        union = int(pred_sizes[p_label] + gt_sizes[j_label] - inter)
        return (inter / union) if union > 0 else 0.0, inter

    results = {"AP_per_iou": {}, "metrics_per_iou": {}}

    # We compute AP for each IoU threshold
    for tau in iou_thresholds:
        tau = float(tau)

        matched_gt = np.zeros(n_gt + 1, dtype=bool)  # 1..n_gt used
        tp_flags = []  # 1 if prediction is TP else 0 in ranking order
        fp_flags = []  # 1 if FP else 0

        for (p_label, score) in pred_instances:
            # find best GT match (unmatched) by IoU
            best_j = 0
            best_iou = 0.0

            #We look for candidates: only the GT labels that overlap
            pred_vox = (pred_lab == p_label)
            overlap_labels = gt_lab[pred_vox]
            if overlap_labels.size == 0:
                # no overlap with any GT -> FP
                tp_flags.append(0)
                fp_flags.append(1)
                continue

            candidates = np.unique(overlap_labels)
            candidates = candidates[candidates != 0]

            for j in candidates:
                if matched_gt[j]:
                    continue
                iou, _ = iou_pred_gt(p_label, j)
                if iou > best_iou:
                    best_iou = iou
                    best_j = int(j)

            if best_j != 0 and best_iou >= tau:
                matched_gt[best_j] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        tp_flags = np.array(tp_flags, dtype=np.float64)
        fp_flags = np.array(fp_flags, dtype=np.float64)

        if tp_flags.size == 0:
            # no predictions at all
            ap = 0.0
            results["AP_per_iou"][tau] = float(ap)
            results["metrics_per_iou"][tau] = {
                "AP": float(ap),
                "n_gt": int(n_gt),
                "n_pred": 0,
                "TP": 0,
                "FP": 0,
                "FN": int(n_gt),
            }
            continue

        tp_cum = np.cumsum(tp_flags)
        fp_cum = np.cumsum(fp_flags)

        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        recalls = tp_cum / float(n_gt)

        # VOC trick: make precision non-increasing when recall increases
        precisions_mono = precisions.copy()
        for i in range(len(precisions_mono) - 2, -1, -1):
            precisions_mono[i] = max(precisions_mono[i], precisions_mono[i + 1])

        # Add (0,1) start for integration stability
        recalls_ext = np.concatenate(([0.0], recalls, [1.0]))
        precisions_ext = np.concatenate(([precisions_mono[0]], precisions_mono, [0.0]))

        ap = float(np.trapz(precisions_ext, recalls_ext))

        TP = int(tp_cum[-1])
        FP = int(fp_cum[-1])
        FN = int(n_gt - matched_gt[1:].sum())

        results["AP_per_iou"][tau] = float(ap)
        results["metrics_per_iou"][tau] = {
            "AP": float(ap),
            "n_gt": int(n_gt),
            "n_pred": int(len(pred_instances)),
            "TP": TP,
            "FP": FP,
            "FN": FN,
        }

    results["mAP"] = float(np.mean(list(results["AP_per_iou"].values())))
    return results


"""
def perform_mAP_from_prob(
    gt_mask,
    pred_prob,
    iou_thresholds=(0.25, 0.5),
    connectivity=26,
    prob_thresholds=None,
    min_cc_size=1,
):

    Lesion-wise mAP from a PROBABILITY MAP.

    Steps:
      - sweep prob thresholds -> binary masks
      - connected components on GT and pred
      - greedy IoU matching -> TP/FP/FN
      - build precision-recall curve
      - AP = integral under PR curve
      - mAP = mean(AP over IoU thresholds)

    Notes:
      - Assumes pred_prob in [0,1] but works if logits too (thresholds then must be adjusted).
      - GT is binarized as gt_mask>0.
  
    if gt_mask.shape != pred_prob.shape:
        raise ValueError(f"Shape mismatch: GT {gt_mask.shape} vs Prob {pred_prob.shape}")

    gt_bin = gt_mask > 0

    # default thresholds: dense enough but not crazy
    if prob_thresholds is None:
        prob_thresholds = np.linspace(0.05, 0.95, 19) #np.linspace(0.05, 0.95, 19)

    # connectivity struct
    if connectivity == 26:
        struct = np.ones((3, 3, 3), dtype=np.int8)
    elif connectivity == 6:
        struct = np.zeros((3, 3, 3), dtype=np.int8)
        struct[1, 1, 1] = 1
        struct[0, 1, 1] = struct[2, 1, 1] = 1
        struct[1, 0, 1] = struct[1, 2, 1] = 1
        struct[1, 1, 0] = struct[1, 1, 2] = 1
    else:
        raise ValueError("connectivity must be 6 or 26")

    gt_lab, n_gt = label(gt_bin, structure=struct)
    gt_sizes = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)

    results = {"AP_per_iou": {}, "metrics_per_iou": {}}

    # edge case: no GT lesions
    if n_gt == 0:
        for tau in iou_thresholds:
            # If no GT, detection AP is defined as 1 only if model predicts nothing at all.
            # We'll compute across thresholds; easiest is:
            # AP=1 if always predicts nothing; else AP=0.
            any_pred = False
            for t in prob_thresholds:
                if np.any(pred_prob > t):
                    any_pred = True
                    break
            ap = 0.0 if any_pred else 1.0
            results["AP_per_iou"][float(tau)] = float(ap)
            results["metrics_per_iou"][float(tau)] = {
                "note": "n_gt=0 special case",
                "n_gt": 0,
            }
        results["mAP"] = float(np.mean(list(results["AP_per_iou"].values())))
        return results

    def filter_small_cc(lab, sizes, min_size):
        cc_keep = sizes >= min_size
        cc_keep[0] = False #we ignore all background
        mask = cc_keep[lab]
        lab_filt,_ = label(mask, structure=struct)
        return lab_filt


    def match_counts(pred_bin, tau):
        pred_lab, n_pred = label(pred_bin, structure=struct)
        pred_sizes = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)
        
        pred_lab = filter_small_cc(pred_lab, pred_sizes, min_cc_size)
        pred_lab, n_pred = label(pred_lab > 0, structure=struct)
        pred_sizes = np.bincount(pred_lab.ravel(), minlength=n_pred+1)

        matched_gt = np.zeros(n_gt + 1, dtype=bool)
        TP = 0
        FP = 0

        for p in range(1, n_pred + 1):
            if pred_sizes[p] == 0:
                continue
            pred_vox = (pred_lab == p)
            gt_overlap_labels = gt_lab[pred_vox]
            if gt_overlap_labels.size == 0:
                FP += 1
                continue

            overlap_counts = np.bincount(gt_overlap_labels, minlength=n_gt + 1)
            overlap_counts[0] = 0

            best_j = 0
            best_iou = 0.0
            for j in np.where(overlap_counts > 0)[0]:
                if matched_gt[j]:
                    continue
                inter = overlap_counts[j]
                union = pred_sizes[p] + gt_sizes[j] - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j != 0 and best_iou >= tau:
                TP += 1
                matched_gt[best_j] = True
            else:
                FP += 1

        FN = int(n_gt - matched_gt[1:].sum())
        return TP, FP, FN, n_pred

    # AP for each IoU tau
    for tau in iou_thresholds:
        tau = float(tau)

        precisions = []
        recalls = []

        for t in prob_thresholds:
            pred_bin = pred_prob > float(t)
            TP, FP, FN, n_pred = match_counts(pred_bin, tau)
            print("n_pred:" , n_pred)
            print("True positive " ,TP, "False Positive " , FP, "False Negative " , FN)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)

        # sort by recall increasing
        recalls = np.array(recalls, dtype=np.float64)
        precisions = np.array(precisions, dtype=np.float64)
        order = np.argsort(recalls)
        recalls = recalls[order]
        precisions = precisions[order]

        # make precision monotonically decreasing (standard AP trick)
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # integrate PR curve (trapz over recall)
        ap = float(np.trapz(precisions, recalls))

        results["AP_per_iou"][tau] = ap
        results["metrics_per_iou"][tau] = {
            "AP": ap,
            "PR_points": len(prob_thresholds),
        }

    results["mAP"] = float(np.mean(list(results["AP_per_iou"].values())))
    return results
"""

def perform_F1(gt_mask, pred_mask, iou_thresholds=(0.25, 0.5), connectivity=26, bin_threshold=0.0):
    
    """
We compare connected components between  ground-truth mask and a predicted mask.

- Both masks are binarized using a configurable threshold.
- Lesions are defined as connected components using either 6- or 26-connectivity.
- Each predicted lesion is matched to at most one ground-truth lesion based on IoU.
- A prediction is counted as a true positive if the IoU exceeds a given threshold.
- Unmatched predictions are counted as false positives.
- Unmatched ground-truth lesions are counted as false negatives.

We repeat these steps for multiple IoU thresholds.
For each threshold, we compute lesion-level precision and recall.
The final score (`F1`) is obtained by averaging the scores across all thresholds.

This metric focuses on lesion detection quality rather than voxel-wise overlap,
which is more appropriate for small and sparse MS lesions.It gives us a comparison ground
with other academic papers that show F1 results rather than mAP
"""
 
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(f"Shape mismatch: GT {gt_mask.shape} vs Pred {pred_mask.shape}")

    gt_bin = (gt_mask > bin_threshold)
    pred_bin = (pred_mask > bin_threshold)

    if connectivity == 26:
        struct = np.ones((3, 3, 3), dtype=np.int8)
    elif connectivity == 6:
        struct = np.zeros((3, 3, 3), dtype=np.int8)
        struct[1, 1, 1] = 1
        struct[0, 1, 1] = struct[2, 1, 1] = 1
        struct[1, 0, 1] = struct[1, 2, 1] = 1
        struct[1, 1, 0] = struct[1, 1, 2] = 1
    else:
        raise ValueError("connectivity must be 6 or 26")

    gt_lab, n_gt = label(gt_bin, structure=struct)
    pred_lab, n_pred = label(pred_bin, structure=struct)

    gt_sizes = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)
    pred_sizes = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)

    results = {"AP_per_iou": {}, "metrics_per_iou": {}}

    if n_gt == 0 and n_pred == 0:
        for tau in iou_thresholds:
            results["AP_per_iou"][float(tau)] = 1.0
            results["metrics_per_iou"][float(tau)] = {
                "precision": 1.0, "recall": 1.0,
                "TP": 0, "FP": 0, "FN": 0,
                "n_gt": 0, "n_pred": 0
            }
        results["F1"] = float(np.mean(list(results["AP_per_iou"].values())))
        return results

    if n_gt == 0 and n_pred > 0:
        for tau in iou_thresholds:
            results["AP_per_iou"][float(tau)] = 0.0
            results["metrics_per_iou"][float(tau)] = {
                "precision": 0.0, "recall": 0.0,
                "TP": 0, "FP": n_pred, "FN": 0,
                "n_gt": 0, "n_pred": n_pred
            }
        results["F1"] = float(np.mean(list(results["AP_per_iou"].values())))
        return results

    pred_labels = list(range(1, n_pred + 1))

    for tau in iou_thresholds:
        tau = float(tau)
        matched_gt = np.zeros(n_gt + 1, dtype=bool)
        #print(matched_gt)
        TP = 0
        FP = 0

        for p in pred_labels:
            pred_vox = (pred_lab == p)
            if pred_sizes[p] == 0:
                continue

            gt_overlap_labels = gt_lab[pred_vox]
            if gt_overlap_labels.size == 0:
                FP += 1
                continue

            overlap_counts = np.bincount(gt_overlap_labels, minlength=n_gt + 1)
            overlap_counts[0] = 0

            best_j = 0
            best_iou = 0.0

            candidates = np.where(overlap_counts > 0)[0]
            #print(candidates)
            for j in candidates:
                if matched_gt[j]:
                    continue
                inter = overlap_counts[j]
                union = pred_sizes[p] + gt_sizes[j] - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j != 0 and best_iou >= tau:
                TP += 1
                matched_gt[best_j] = True
            else:
                FP += 1
                
        FN = int(n_gt - matched_gt[1:].sum())
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / n_gt if n_gt > 0 else 0.0
        AP_tau = precision * recall

        results["AP_per_iou"][tau] = float(AP_tau)
        results["metrics_per_iou"][tau] = {
            "precision": float(precision),
            "recall": float(recall),
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "n_gt": int(n_gt),
            "n_pred": int(n_pred),
        }

    results["F1"] = float(np.mean(list(results["AP_per_iou"].values())))
    return results



def list_case_ids_from_labels(labels_dir: str):
    """
    nnU-Net labels convention: CASEID.nii.gz (or .nii)
    Return list of case_ids (filename without extension).
    """
    case_ids = []
    for f in sorted(os.listdir(labels_dir)):
        if f.endswith(".nii.gz"):
            case_ids.append(f[:-7])  # remove ".nii.gz"
        elif f.endswith(".nii"):
            case_ids.append(f[:-4])  # remove ".nii"
    return case_ids


def find_case_mask(folder: str, case_id: str):
    """
    Find CASEID.nii.gz or CASEID.nii in folder. Return path or None.
    """
    p1 = os.path.join(folder, f"{case_id}.nii.gz")
    p2 = os.path.join(folder, f"{case_id}.nii")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None

def find_case_npz(folder: str, case_id: str):
    """
    Find CASEID.npz in folder. Return path or None.
    """
    p = os.path.join(folder, f"{case_id}.npz")
    if os.path.exists(p):
        return p
    return None


def compute_stats(scores):
    """
    scores: list[float]
    returns dict with mean/median/min/max/var/std and n
    """
    arr = np.array(scores, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else None,
        "median": float(np.median(arr)) if arr.size else None,
        "min": float(np.min(arr)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
        "var": float(np.var(arr)) if arr.size else None,
        "std": float(np.std(arr)) if arr.size else None,
    }


def print_summary_block(title: str, results_dict: dict):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for model_name, payload in results_dict.items():
        if model_name == "meta":
            continue

        stats = payload.get("stats", {})
        print(f"\n[{model_name}]")
        print(f"  n     : {stats.get('n')}")
        print(f"  mean  : {stats.get('mean')}")
        print(f"  median: {stats.get('median')}")
        print(f"  min   : {stats.get('min')}")
        print(f"  max   : {stats.get('max')}")
        print(f"  var   : {stats.get('var')}")
        print(f"  std   : {stats.get('std')}")

    meta = results_dict.get("meta", {})
    if meta.get("missing"):
        print("\n[Missing predictions]")
        for model_name, cases in meta["missing"].items():
            print(f"  {model_name}: {len(cases)} missing (first 10: {cases[:10]})")

def resolve_pred_case_id(gt_case_id: str, dataset_id: int) -> str:
    """
    Map GT case_id to predicted mask filename (without extension).

    421: GT = SEP_094        -> PRED = SEP_094
    422: GT = MICCAI2016_01047   -> PRED = MICCAI_01047
    423: GT = OPENMS longitudinal 
    """
    if dataset_id == 421:
        # GT already like SEP_094
        return gt_case_id

    elif dataset_id == 422:
        # GT already like MICCAI_01047
        return gt_case_id
        # GT already like OPENMS longitudinal 
    elif dataset_id == 423:
        return gt_case_id
    
    elif dataset_id == 424:
         return gt_case_id   

    else:
        raise ValueError(f"Unsupported dataset_id: {dataset_id}")


def find_case_prob(folder: str, case_id: str):
    p1 = os.path.join(folder, f"{case_id}_prob.nii.gz")
    p2 = os.path.join(folder, f"{case_id}_prob.nii")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None

def load_probmap_any(path: str, fg_channel: int = 1) -> np.ndarray:
    """
    Load probability map from either:
      - nnU-Net .npz (contains probabilities/softmax)
      - NIfTI .nii/.nii.gz probmap

    Returns prob map as float32 array (same shape as GT).
    """
    if path.endswith(".npz"):
        data = np.load(path)
        if "probabilities" in data:
            arr = data["probabilities"]
        elif "softmax" in data:
            arr = data["softmax"]
        else:
            raise KeyError(f"{path} has no 'probabilities' or 'softmax'. keys={list(data.keys())}")

        # arr can be (C, Z, Y, X) or (Z,Y,X) in some cases
        if arr.ndim == 4:
            if arr.shape[0] == 1:
                prob = arr[0]
            else:
                prob = arr[fg_channel]
        elif arr.ndim == 3:
            prob = arr
        else:
            raise ValueError(f"Unexpected npz prob shape: {arr.shape}")

        return prob.astype(np.float32)

    # NIfTI case
    prob = nib.load(path).get_fdata()
    if prob.ndim == 4 and prob.shape[-1] == 1:
        prob = prob[..., 0]
    if prob.ndim != 3:
        raise ValueError(f"Unexpected NIfTI prob shape: {prob.shape}")
    return prob.astype(np.float32)



#After shape mismatch problem, a fix is provided:
def align_prob_to_gt(gt, prob, case_id = ""):
    """
    Ensures prob has same shape as gt by trying all permutations around a common axis
    gt : (D, H ,W) => We permute prob (D,H,W) until the shapes match.
    This function has been added as some  rare cases have different orientation.
    
    """
    #straight forward approach
    if gt.shape == prob.shape:
        return prob
    
    #try common nnUnet cases
    candidates = [(2, 0, 1), #(H,W,D) = > (D,H,W)
                  (1, 2, 0), #(W,D,H) = > (D,H,W)
                  (0, 2 ,1), #(D,W,H) = > (D,H,W)
                  (2, 1, 0),
                  (1, 0, 2),
                  (0, 1, 2)
                  ] 
    for axes in candidates:
        p = np.transpose(prob, axes)
        if p.shape == gt.shape:
            print(f"[INFO] We transposed prob for {case_id} using {axes}")
            return p
    raise ValueError(f" Can't align probmap to GT for {case_id} using axes {axes}"
                     f" GT shape = {gt.shape} , prob shape = {prob.shape}")

def main():
    #all the extensions are created here
    parser = argparse.ArgumentParser(description="Simple benchmarking for nnU-Net-format masks: Dice / VS / mAP")
    parser.add_argument("--labelsTs", required=True, help="Path to GT masks folder (nnU-Net format), e.g. labelsTs/")
    parser.add_argument("--models", required=True,
                        help="JSON string or JSON file path mapping model_name -> prediction_folder.\n"
                             "Example JSON string: '{\"nnunet\":\"/preds/nnunet\",\"segresnet\":\"/preds/segresnet\",\"lst\":\"/preds/lst\"}'")
    parser.add_argument("--threshold", type=float, default=0.0, help="Binarization threshold (>threshold)")
    parser.add_argument("--iou_thresholds", default="0.25,0.5", help="mAP IoU thresholds, e.g. '0.25,0.5'")
    parser.add_argument("--connectivity", type=int, default=26, help="mAP connectivity: 6 or 26")
    parser.add_argument("--save_json", default="", help="Optional output folder to save results JSON files")
    parser.add_argument("--dataset_id", type=int, required=True, choices=[421, 422,423,424], help="Dataset ID to resolve prediction naming (421=SEP, 422=MICCAI)")
    
    
    
    args = parser.parse_args()
	
    labelsTs = args.labelsTs
    if not os.path.isdir(labelsTs):
        raise FileNotFoundError(f"labelsTs folder not found: {labelsTs}")

    # Parse model mapping
    models_arg = args.models
    model_map = None
    if os.path.exists(models_arg) and models_arg.endswith(".json"):
        with open(models_arg, "r", encoding="utf-8") as f:
            model_map = json.load(f)
    else:
        model_map = json.loads(models_arg)

    if not isinstance(model_map, dict) or len(model_map) == 0:
        raise ValueError("models must map model_name -> prediction_folder")

    for k, v in model_map.items():
        if not os.path.isdir(v):
            raise FileNotFoundError(f"Prediction folder for model '{k}' not found: {v}")

    iou_thresholds = tuple(float(x.strip()) for x in args.iou_thresholds.split(",") if x.strip() != "")

    case_ids = list_case_ids_from_labels(labelsTs)
    if len(case_ids) == 0:
        raise RuntimeError(f"No label files found in {labelsTs}")

    # Results structures
    DICE_results_dict = {"meta": {"missing": {}}}
    VS_results_dict = {"meta": {"missing": {}}}
    mAP_results_dict = {"meta": {"missing": {}}}
    F1_results_dict = {"meta": {"missing": {}}}

    # Init per model
    for model_name in model_map.keys():
        #we initialize all the keys that are going to store the results.
        DICE_results_dict[model_name] = {"scores": [], "per_case": []}
        VS_results_dict[model_name] = {"scores": [], "per_case": [], "V_gt": [], "V_pred": []}
        mAP_results_dict[model_name] = {"scores": [], "per_case": [], "AP_per_iou": {}}
        F1_results_dict[model_name] = {"scores": [], "per_case": [], "AP_per_iou": {}}
        DICE_results_dict["meta"]["missing"][model_name] = []
        VS_results_dict["meta"]["missing"][model_name] = []
        mAP_results_dict["meta"]["missing"][model_name] = []
        F1_results_dict["meta"]["missing"][model_name] = []
    # Loop cases
    for cid in case_ids:
        gt_path = find_case_mask(labelsTs, cid)
        if gt_path is None:
            # Should not happen since cid list comes from labelsTs
            continue

        for model_name, pred_dir in model_map.items():
            pred_case_id = resolve_pred_case_id(cid, args.dataset_id)
            pred_path = find_case_mask(pred_dir, pred_case_id)
            pred_prob_path = find_case_prob(pred_dir, pred_case_id)
            gt_img = nib.load(gt_path).get_fdata().astype(np.float32)
            
            #this is a specificity for nnU-Net whih stores probmaps as .npz files 
            if model_name=="nnunet":
                print("[DEBUG] Looking for", os.path.join(pred_dir, f"{pred_case_id}.npz"))
                print("[DEBUG] exists ? ", os.path.exists(os.path.join(pred_dir, f"{pred_case_id}.npz")))
                #if we are retrieving nnunet predictions
                print("[DEBUG] GT cid =", cid, "| pred_case_id =", pred_case_id)
                print("[DEBUG] expecting npz =", os.path.join(pred_dir, f"{pred_case_id}.npz"))
                pred_prob_path = find_case_npz(pred_dir,pred_case_id)
                d = np.load(pred_prob_path)
                print("NPZ keys: ", list(d.keys()))
                prob = d["probabilities"]
                print("probabilities shape: ", prob.shape, "dtype", prob.dtype, "min/max ", prob.min(), prob.max())
                pred_prob_lesion = prob[1]
                #pred_prob_lesion = np.transpose(pred_prob_lesion, (2,0,1)) #H W D to D H W
                pred_prob_lesion = align_prob_to_gt(gt_img, pred_prob_lesion, case_id = cid)
            
            if pred_path is None:
                DICE_results_dict["meta"]["missing"][model_name].append(cid)
                VS_results_dict["meta"]["missing"][model_name].append(cid)
                mAP_results_dict["meta"]["missing"][model_name].append(cid)
                F1_results_dict["meta"]["missing"][model_name].append(cid)
                continue

            try:
                gt, pred = load_masks(gt_path, pred_path)
                print("segresnet: gt shape: ", gt.shape, "pred shape: " , pred.shape)
                # Dice
                dice = perform_dice(gt, pred, threshold=args.threshold)
                DICE_results_dict[model_name]["scores"].append(dice)
                DICE_results_dict[model_name]["per_case"].append((cid, dice))

                # VS
                vs, V_gt, V_pred, abs_diff = perform_volume_similarity(gt, pred, threshold=args.threshold)
                VS_results_dict[model_name]["scores"].append(vs)
                VS_results_dict[model_name]["per_case"].append((cid, vs, V_gt, V_pred, abs_diff))
                VS_results_dict[model_name]["V_gt"].append(V_gt)
                VS_results_dict[model_name]["V_pred"].append(V_pred)

                # mAP
                print("GT", gt_img.shape, "pred prob ", pred_prob_lesion.shape)
                if gt_img.ndim == 4 and gt_img.shape[-1] == 1:
                    gt_img = gt_img[..., 0]

                prob_img = load_probmap_any(pred_prob_path, fg_channel=1)
                #prob_img = nib.load(pred_prob_path).get_fdata() #loads with either npz or nii
                
                if model_name == "nnunet":
                    
                    m = perform_mAP_from_prob(
                        gt_mask=gt_img,pred_prob=pred_prob_lesion,
                        iou_thresholds=iou_thresholds, 
                        connectivity=args.connectivity,
                        binarize_threshold=0.05,      # fixed threshold for creatign the connected components
                        min_cc_size=20,               # the minimum voxel size of a connected component
                        score_mode="max",             # lesion score = max prob
                        )
                    
    
                else:
                    m = perform_mAP_from_prob(
                        gt_mask=gt_img,pred_prob=prob_img,
                        iou_thresholds=iou_thresholds,
                        connectivity=args.connectivity,
                        binarize_threshold=0.05,     
                        min_cc_size=20,               
                        score_mode="max",
                        )
                    

                
                mAP = float(m["mAP"])
                mAP_results_dict[model_name]["scores"].append(mAP)
                mAP_results_dict[model_name]["per_case"].append((cid, mAP))
                print(model_name, mAP_results_dict[model_name]["per_case"])
                # Optional: aggregate AP per IoU (mean across cases)
                for tau, ap in m["AP_per_iou"].items():
                    tau = float(tau)
                    if tau not in mAP_results_dict[model_name]["AP_per_iou"]:
                        mAP_results_dict[model_name]["AP_per_iou"][tau] = []
                    mAP_results_dict[model_name]["AP_per_iou"][tau].append(float(ap))
                    
                    
                m = perform_F1(
                    gt, pred,
                    iou_thresholds=iou_thresholds,
                    connectivity=args.connectivity,
                    bin_threshold=args.threshold
                )
                
                F1 = float(m["F1"])
                F1_results_dict[model_name]["scores"].append(F1)
                F1_results_dict[model_name]["per_case"].append((cid, F1))
                print(model_name, F1_results_dict[model_name]["per_case"])
                # Optional: aggregate AP per IoU (mean across cases)
                for tau, ap in m["AP_per_iou"].items():
                    tau = float(tau)
                    if tau not in F1_results_dict[model_name]["AP_per_iou"]:
                        F1_results_dict[model_name]["AP_per_iou"][tau] = []
                    F1_results_dict[model_name]["AP_per_iou"][tau].append(float(ap))

            except Exception as e:
                # Treat as missing/failed case 
                DICE_results_dict["meta"]["missing"][model_name].append(f"{cid} (error: {e})")
                VS_results_dict["meta"]["missing"][model_name].append(f"{cid} (error: {e})")
                mAP_results_dict["meta"]["missing"][model_name].append(f"{cid} (error: {e})")
                F1_results_dict["meta"]["missing"][model_name].append(f"{cid} (error: {e})")
                continue

    # Compute stats per model
    for model_name in model_map.keys():
        DICE_results_dict[model_name]["stats"] = compute_stats(DICE_results_dict[model_name]["scores"])
        VS_results_dict[model_name]["stats"] = compute_stats(VS_results_dict[model_name]["scores"])
        mAP_results_dict[model_name]["stats"] = compute_stats(mAP_results_dict[model_name]["scores"])
        mAP_results_dict[model_name]["stats"] = compute_stats(mAP_results_dict[model_name]["scores"])
        F1_results_dict[model_name]["stats"] = compute_stats(F1_results_dict[model_name]["scores"])
        # Extra VS global volume stats (simple)
        if len(VS_results_dict[model_name]["V_gt"]) > 0:
            VS_results_dict[model_name]["volume_stats"] = {
                "mean_V_gt_vox": float(np.mean(VS_results_dict[model_name]["V_gt"])),
                "mean_V_pred_vox": float(np.mean(VS_results_dict[model_name]["V_pred"])),
            }
        else:
            VS_results_dict[model_name]["volume_stats"] = {"mean_V_gt_vox": None, "mean_V_pred_vox": None}

        # Extra mAP per IoU (mean)
        if mAP_results_dict[model_name]["AP_per_iou"]:
            mAP_results_dict[model_name]["AP_per_iou_mean"] = {
                float(tau): float(np.mean(vals)) for tau, vals in mAP_results_dict[model_name]["AP_per_iou"].items()
            }
        else:
            mAP_results_dict[model_name]["AP_per_iou_mean"] = {}
            
        
        if F1_results_dict[model_name]["AP_per_iou"]:
            F1_results_dict[model_name]["AP_per_iou_mean"] = {
                float(tau): float(np.mean(vals)) for tau, vals in F1_results_dict[model_name]["AP_per_iou"].items()
            }
        else:
            F1_results_dict[model_name]["AP_per_iou_mean"] = {}

    # Print results 
    print_summary_block("DICE RESULTS", DICE_results_dict)
    print_summary_block("VOLUME SIMILARITY RESULTS", VS_results_dict)

    # Print VS extra volumes
    print("\n" + "=" * 70)
    print("VS VOLUME STATS (mean volumes in voxels)")
    print("=" * 70)
    for model_name in model_map.keys():
        vs_vol = VS_results_dict[model_name].get("volume_stats", {})
        print(f"{model_name}: mean_V_gt_vox={vs_vol.get('mean_V_gt_vox')} | mean_V_pred_vox={vs_vol.get('mean_V_pred_vox')}")

    # Print mAP with AP per IoU mean
    print("\n" + "=" * 70)
    print("mAP RESULTS")
    print("=" * 70)
    for model_name in model_map.keys():
        stats = mAP_results_dict[model_name]["stats"]
        print(f"\n[{model_name}]")
        print(f"  n     : {stats.get('n')}")
        print(f"  mean  : {stats.get('mean')}")
        print(f"  median: {stats.get('median')}")
        print(f"  min   : {stats.get('min')}")
        print(f"  max   : {stats.get('max')}")
        print(f"  var   : {stats.get('var')}")
        print(f"  std   : {stats.get('std')}")
        print(f"  AP_per_iou_mean: {mAP_results_dict[model_name].get('AP_per_iou_mean', {})}")
#F1 scores
    print("\n" + "=" * 70)
    print("F1 RESULTS")
    print("=" * 70)
    for model_name in model_map.keys():
        stat2 = F1_results_dict[model_name]["stats"]
        print(f"\n[{model_name}]")
        print(f"  n     : {stat2.get('n')}")
        print(f"  mean  : {stat2.get('mean')}")
        print(f"  median: {stat2.get('median')}")
        print(f"  min   : {stat2.get('min')}")
        print(f"  max   : {stat2.get('max')}")
        print(f"  var   : {stat2.get('var')}")
        print(f"  std   : {stat2.get('std')}")
        print(f"  AP_per_iou_mean: {F1_results_dict[model_name].get('AP_per_iou_mean', {})}")
        
        

    # Optional save JSONs
    if args.save_json:
        os.makedirs(args.save_json, exist_ok=True)
        with open(os.path.join(args.save_json, "DICE_results.json"), "w", encoding="utf-8") as f:
            json.dump(DICE_results_dict, f, indent=2)
        with open(os.path.join(args.save_json, "VS_results.json"), "w", encoding="utf-8") as f:
            json.dump(VS_results_dict, f, indent=2)
        with open(os.path.join(args.save_json, "mAP_results.json"), "w", encoding="utf-8") as f:
            json.dump(mAP_results_dict, f, indent=2)
        with open(os.path.join(args.save_json, "F1_results.json"), "w", encoding="utf-8") as f:
            json.dump(F1_results_dict, f, indent=2)            
        
        print(f"\n Saved JSON results to: {args.save_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()

