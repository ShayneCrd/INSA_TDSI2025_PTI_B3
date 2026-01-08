#!/usr/bin/env bash
set -euo pipefail  # strict mode

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset_id> [folds]"
  echo "  dataset_id: 421 | 422 | 423 | 424"
  echo "  folds (optional): e.g. \"0 1 2 3 4\" (default: \"0 1 2 3 4\")"
  exit 1
fi

DATASET_NUM="$1"
FOLDS="${2:-"0 1 2 3 4"}"

# ==========================================
# Resolve paths
# this script should be located in SegResNetFrame/
# Repo root = parent directory of SegResNetFrame/
# ==========================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATASET_ID="Dataset${DATASET_NUM}_TDSI2025"

NNUNET_RAW="${REPO_ROOT}/nnUnetFrame/nnUNet_raw/${DATASET_ID}"
SPLITS="${REPO_ROOT}/nnUnetFrame/nnUNet_preprocessed/${DATASET_ID}/splits_final.json"
OUT_ROOT="${REPO_ROOT}/SegResNetFrame/SegResNet_results/cross_validation_training/${DATASET_ID}"
#OUT_ROOT="${REPO_ROOT}/${DATASET_ID}"

# ==========================================
# Hyperparams per dataset, we observed a different lr required for 422
# ==========================================
case "${DATASET_NUM}" in
  421)
    MODALITIES="0,1,2"
    EPOCHS=300
    PATCH="128,128,32"
    BATCH_SIZE=1
    LR="1e-4"
    ;;
  422)
    MODALITIES="0,1,2"      
    EPOCHS=300              
    PATCH="128,128,32"      
    BATCH_SIZE=1            
    LR="5e-4"               
    ;;
  423)
    MODALITIES="0,1,2,3,4,5,6,7,8"    
    EPOCHS=300              
    PATCH="128,128,32"      
    BATCH_SIZE=1            
    LR="1e-4"               
    ;;
  424)
    MODALITIES="0,1,2,3"      
    EPOCHS=300              
    PATCH="128,128,32"     
    BATCH_SIZE=1            
    LR="1e-4"               
    ;;
  *)
    echo "ERROR: dataset_id must be one of: 421 422 423 424"
    exit 1
    ;;
esac

# ==========================================
# Sanity checks (we forgot it in last version)
# ==========================================
echo "========================================"
echo "[INFO] Repo root   : ${REPO_ROOT}"
echo "[INFO] Dataset     : ${DATASET_ID}"
echo "[INFO] nnUNet raw  : ${NNUNET_RAW}"
echo "[INFO] splits file : ${SPLITS}"
echo "[INFO] out root    : ${OUT_ROOT}"
echo "[INFO] folds       : ${FOLDS}"
echo "========================================"

[[ -d "${NNUNET_RAW}" ]] || { echo "ERROR: missing NNUNET_RAW dir: ${NNUNET_RAW}"; exit 1; }
[[ -f "${SPLITS}" ]] || { echo "ERROR: missing splits file: ${SPLITS} (run nnUNet plan+preprocess first)"; exit 1; }

mkdir -p "${OUT_ROOT}"

# Optional: ensure we run from SegResNetFrame so relative python import/paths behave
cd "${SCRIPT_DIR}"

# ==========================================
# Train loop (just launches train_SegResNet.py)
# ==========================================
for FOLD in ${FOLDS}; do
  echo "========================================"
  echo "[INFO] Training SegResNet | dataset=${DATASET_NUM} | fold=${FOLD}"
  echo "========================================"

  OUT_DIR="${OUT_ROOT}/fold_${FOLD}"
  mkdir -p "${OUT_DIR}"

  python train_SegResNet.py \
    --nnunet_raw "${NNUNET_RAW}" \
    --out "${OUT_DIR}" \
    --use_splits "${SPLITS}" \
    --fold "${FOLD}" \
    --modalities "${MODALITIES}" \
    --epochs "${EPOCHS}" \
    --patch "${PATCH}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    2>&1 | tee "${OUT_DIR}/train_fold_${FOLD}.log"

  echo "[INFO] Fold ${FOLD} finished."
done

