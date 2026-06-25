#!/usr/bin/env bash
# ── Конфиг для генома человека (NVIDIA RTX A5000, 24 ГБ, bert-base) ──────────
VARIANT="${1:-blastp_enformer}"

GENA_ROOT="${HOME}/GENA_LM_BASE"
DATA_ROOT="${HOME}/human/hdf5/${VARIANT}"
MODEL_PATH="${GENA_ROOT}/runs/human_${VARIANT}"
MODEL_CHECKPOINT="${GENA_ROOT}/model_checkpoint_base"
INIT_CHECKPOINT="${GENA_ROOT}/model_checkpoint_base/pytorch_model.bin"
PYTHON="/home/chumanova/.conda/envs/gena_lm/bin/python"

export PYTHONPATH="${GENA_ROOT}/model_checkpoint_base:${PYTHONPATH}"

mkdir -p "${MODEL_PATH}"
LOG_FILE="${MODEL_PATH}/run_$(date +%Y%m%d_%H%M%S).log"

echo "Старт: $(date)"           | tee "${LOG_FILE}"
echo "Вариант: human_${VARIANT}" | tee -a "${LOG_FILE}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "${LOG_FILE}"
echo "Init: ${INIT_CHECKPOINT}" | tee -a "${LOG_FILE}"

# Имена файлов у человека содержат _human_
TRAIN="${DATA_ROOT}/${VARIANT}_human_promoters_train.h5"
VALID="${DATA_ROOT}/${VARIANT}_human_promoters_valid.h5"
TEST="${DATA_ROOT}/${VARIANT}_human_promoters_test.h5"

for f in "$TRAIN" "$VALID" "$TEST"; do
    if [ ! -f "$f" ]; then
        echo "❌ Файл не найден: $f"
        exit 1
    fi
done
echo "✅ Все датасеты найдены" | tee -a "${LOG_FILE}"

"$PYTHON" "${HOME}/notebooks/GENA_LM/run_finetuning_all_species.py" \
    --data_path        "$TRAIN" \
    --valid_data_path  "$VALID" \
    --test_data_path   "$TEST" \
    --model_path       "$MODEL_PATH" \
    --init_checkpoint  "$INIT_CHECKPOINT" \
    --tokenizer        "$MODEL_CHECKPOINT" \
    --model_cfg        "$MODEL_CHECKPOINT/config.json" \
    --model_cls        "modeling_bert:BertForSequenceClassification" \
    --input_seq_len    512 \
    --batch_size       16 \
    --iters            200000 \
    --lr               1e-5 \
    --weight_decay     0.01 \
    --num_warmup_steps 1000 \
    --gradient_accumulation_steps 16 \
    --body_lr_multiplier 0.1 \
    --clip_grad_norm   1.0 \
    --log_interval     100 \
    --valid_interval   500 \
    --early_stopping_patience 9999 \
    --data_n_workers   4 \
    --seed             42 \
    --start_step       0 \
    2>&1 | tee -a "${LOG_FILE}"
