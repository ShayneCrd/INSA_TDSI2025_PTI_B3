#!/usr/bin/env bash
set -e  # stop if any command fails

# ==============================
# CONFIG
# ==============================
NNUNET_RAW="/local/scardell/nnUnetFrame/nnUNet_raw/Dataset421_TDSI2025"
SPLITS="/local/scardell/nnUnetFrame/nnUNet_preprocessed/Dataset421_TDSI2025/splits_final.json"
OUT_ROOT="/local/scardell/SegResNetFrame/SegResNet_results/cross_validation_training/Dataset421"


set -euo pipefail


MODALITIES="0,1,2"
EPOCHS=300
PATCH="128,128,32"
BATCH_SIZE=1
LR="1e-4"

# ==============================
# TRAIN LOOP
# ==============================
for FOLD in 4
do
    echo "========================================"
    echo "[INFO] Training SegResNet | fold=${FOLD}"
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

echo "========================================"
echo "[DONE] Cross-validation training finished"
echo "========================================"

