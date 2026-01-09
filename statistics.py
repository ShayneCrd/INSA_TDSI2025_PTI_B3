#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
statistic.py

We run paired Wilcoxon signed-rank tests (non-parametric) on our benchmarking outputs.

Goal (for examiners):
- For each dataset (Dataset421 -> Dataset424),
- For each metric (DICE / VS / mAP),
- We compare method A = nnUNet against method B = SegResNet and LST-AI (paired per case).

Input:
  benchmark_results/
    Dataset421/
      DICE_results.json
      VS_results.json
      mAP_results.json
    Dataset422/
      ...
    Dataset423/
      ...
    Dataset424/
      ...

Expected JSON structure (from our benchmark.py):
  {
    "meta": {"missing": {...}},
    "nnunet": {"scores": [...], "per_case": [[case_id, score], ...], "stats": {...}},
    "segresnet": {...},
    "lst": {...},
    ...
  }

We:
- Load per-case scores for each model.
- Align pairs by intersecting case_ids (strict pairing).
- Run Wilcoxon signed-rank test with alternative="greater" (H1: nnUNet > B).
- Report: n, median(A), median(B), median(A-B), Wilcoxon statistic, p-value.
- Save a global CSV summary to benchmark_results/wilcoxon_summary.csv

Usage:
  python statistic.py --root /local/scardell/benchmark_results
"""

import os
import json
import argparse
import csv
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception as e:
    raise RuntimeError(
        "scipy is required for Wilcoxon tests. Install it in your env: pip install scipy"
    ) from e


# -----------------------------
# Utilities: IO and parsing
# -----------------------------

METRIC_FILES = {
    "DICE": "DICE_results.json",
    "VS": "VS_results.json",
    "mAP": "mAP_results.json",
}


def safe_load_json(path: str) -> Optional[dict]:
    """We load a JSON file and return None if it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_model_key(k: str) -> str:
    """
    We normalize model keys because our benchmark outputs may use:
      nnunet / nnUNet / nnUnet
      segresnet / SegResNet
      lst / LST / lst_ai
    """
    k0 = k.strip().lower()
    if k0 in {"nnunet", "nnunetv2", "nnunet_v2", "nnunet2"}:
        return "nnunet"
    if k0 in {"segresnet", "seg_resnet", "seg-resnet"}:
        return "segresnet"
    if k0 in {"lst", "lst_ai", "lstai", "lst-ai"}:
        return "lst"
    return k0


def extract_per_case_scores(metric_json: dict) -> Dict[str, Dict[str, float]]:
    """
    We convert the benchmark JSON into:
      scores[model][case_id] = score

    We only use per_case because it preserves pairing information.
    """
    out: Dict[str, Dict[str, float]] = {}
    for k, payload in metric_json.items():
        if k == "meta":
            continue
        mk = normalize_model_key(k)
        per_case = payload.get("per_case", [])
        model_scores: Dict[str, float] = {}
        for row in per_case:
            # per_case rows vary by metric:
            #   Dice: [cid, dice]
            #   mAP : [cid, map]
            #   VS  : [cid, vs, V_gt, V_pred, abs_diff]
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            cid = str(row[0])
            try:
                score = float(row[1])
            except Exception:
                continue
            model_scores[cid] = score
        out[mk] = model_scores
    return out


def paired_arrays(
    a_scores: Dict[str, float],
    b_scores: Dict[str, float],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    We build paired vectors A and B by intersecting case IDs.
    This ensures the test is paired case-by-case (same patients/cases).
    """
    common = sorted(set(a_scores.keys()) & set(b_scores.keys()))
    A = np.array([a_scores[c] for c in common], dtype=np.float64)
    B = np.array([b_scores[c] for c in common], dtype=np.float64)
    return common, A, B


# -----------------------------
# Wilcoxon testing logic
# -----------------------------

def wilcoxon_nnUNet_vs(A: np.ndarray,B: np.ndarray,alternative: str = "greater") -> dict:
    """
    We run the Wilcoxon signed-rank test on paired samples (A vs B).

    Important:
    - We test H1: A > B by default ("greater"), because we want to know whether nnUNet
      is statistically better than the competitor.
    - Wilcoxon ignores exact zeros in differences by default (zero_method="wilcox").
      If all differences are zero, scipy may raise; we handle that gracefully.
    """
    if A.shape != B.shape:
        raise ValueError(f"Paired arrays must match in shape, got A{A.shape} vs B{B.shape}")

    diffs = A - B
    n = int(A.size)

    # If all diffs are exactly zero, we cannot reject H0 and the statistic is not meaningful.
    if n == 0:
        return {"n": 0, "stat": None, "p": None, "note": "no paired cases"}
    if np.allclose(diffs, 0.0):
        return {
            "n": n,
            "stat": 0.0,
            "p": 1.0,
            "note": "all paired differences are zero",
        }

    try:
        res = wilcoxon(A, B, alternative=alternative, zero_method="wilcox", correction=False, mode="auto")
        stat = float(res.statistic)
        p = float(res.pvalue)
        return {"n": n, "stat": stat, "p": p, "note": ""}
    except Exception as e:
        # We still return descriptive stats even if the test fails (e.g., too many zeros).
        return {"n": n, "stat": None, "p": None, "note": f"wilcoxon failed: {e}"}


def summarize_pair(A: np.ndarray, B: np.ndarray) -> dict:
    """We compute robust descriptive stats for A and B, plus A-B."""
    diffs = A - B
    return {
        "median_A": float(np.median(A)) if A.size else None,
        "median_B": float(np.median(B)) if B.size else None,
        "median_diff": float(np.median(diffs)) if diffs.size else None,
        "mean_A": float(np.mean(A)) if A.size else None,
        "mean_B": float(np.mean(B)) if B.size else None,
        "mean_diff": float(np.mean(diffs)) if diffs.size else None,
    }


# -----------------------------
# Main traversal
# -----------------------------

def list_dataset_dirs(root: str) -> List[str]:
    """We find Dataset421..Dataset424 subfolders under the benchmark_results root."""
    ds = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.lower().startswith("dataset"):
            ds.append(p)
    return ds


def dataset_id_from_dir(path: str) -> str:
    return os.path.basename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to benchmark_results (contains Dataset421..Dataset424 subfolders)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold (default 0.05)")
    ap.add_argument("--alternative", default="greater",choices=["greater", "less", "two-sided"], help="Wilcoxon alternative hypothesis (default: greater meaning nnUNet > other)")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"benchmark_results root not found: {root}")

    dataset_dirs = list_dataset_dirs(root)
    if not dataset_dirs:
        raise RuntimeError(f"No DatasetXXX folders found under: {root}")

    # We accumulate rows for a single summary CSV across all datasets/metrics/comparisons.
    summary_rows = []

    print("\n" + "=" * 80)
    print("WILCOXON PAIRED TESTS (A = nnUNet)")
    print("=" * 80)
    print(f"[INFO] root       : {root}")
    print(f"[INFO] alpha      : {args.alpha}")
    print(f"[INFO] alternative: {args.alternative}")
    print("")

    for ds_dir in dataset_dirs:
        ds_name = dataset_id_from_dir(ds_dir)
        print("\n" + "-" * 80)
        print(f"Dataset: {ds_name}")
        print("-" * 80)

        for metric_name, fname in METRIC_FILES.items():
            fpath = os.path.join(ds_dir, fname)
            data = safe_load_json(fpath)
            if data is None:
                print(f"[WARN] Missing {fname} in {ds_name} -> skipping {metric_name}")
                continue

            scores = extract_per_case_scores(data)

            # We require nnunet to exist to run comparisons.
            if "nnunet" not in scores:
                print(f"[WARN] No nnunet key in {fname} -> cannot compare for {metric_name}")
                continue

            a_scores = scores["nnunet"]

            # We compare nnUNet vs SegResNet, then nnUNet vs LST if available.
            comparisons = [("segresnet", "SegResNet"), ("lst", "LST-AI")]

            print(f"\nMetric: {metric_name} ({fname})")

            for b_key, b_label in comparisons:
                if b_key not in scores:
                    print(f"  [INFO] competitor missing: {b_label} (no key '{b_key}') -> skipped")
                    continue

                b_scores = scores[b_key]

                case_ids, A, B = paired_arrays(a_scores, b_scores)
                test = wilcoxon_nnUNet_vs(A, B, alternative=args.alternative)
                desc = summarize_pair(A, B)

                # Decision: significant or not
                significant = (test["p"] is not None) and (test["p"] < args.alpha)

                print(f"  A=nnUNet vs B={b_label}")
                print(f"    paired_n     : {test['n']}")
                print(f"    median(A)    : {desc['median_A']}")
                print(f"    median(B)    : {desc['median_B']}")
                print(f"    median(A-B)  : {desc['median_diff']}")
                print(f"    wilcoxon_stat: {test['stat']}")
                print(f"    p_value      : {test['p']}")
                if test["note"]:
                    print(f"    note         : {test['note']}")
                print(f"    significant  : {significant}")

                summary_rows.append(
                    {
                        "dataset": ds_name,
                        "metric": metric_name,
                        "A_model": "nnUNet",
                        "B_model": b_label,
                        "paired_n": test["n"],
                        "median_A": desc["median_A"],
                        "median_B": desc["median_B"],
                        "median_A_minus_B": desc["median_diff"],
                        "mean_A": desc["mean_A"],
                        "mean_B": desc["mean_B"],
                        "mean_A_minus_B": desc["mean_diff"],
                        "wilcoxon_stat": test["stat"],
                        "p_value": test["p"],
                        "alternative": args.alternative,
                        "alpha": args.alpha,
                        "significant": significant,
                        "note": test["note"],
                    }
                )

    # We save a CSV summary so results can be inserted into the report
    out_csv = os.path.join(root, "wilcoxon_summary.csv")
    fieldnames = [
        "dataset",
        "metric",
        "A_model",
        "B_model",
        "paired_n",
        "median_A",
        "median_B",
        "median_A_minus_B",
        "mean_A",
        "mean_B",
        "mean_A_minus_B",
        "wilcoxon_stat",
        "p_value",
        "alternative",
        "alpha",
        "significant",
        "note",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print("\n" + "=" * 80)
    print("[DONE] Wilcoxon summary written to:")
    print(f"  {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()


