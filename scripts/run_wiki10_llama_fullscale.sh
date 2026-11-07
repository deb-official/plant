#!/bin/bash
# Full-scale PLANT run: Wiki10-31K + Llama-3.1-8B-Instruct
# Follows README_EURLEX_WIKI10.md "Full-Scale Runs" section verbatim,
# parameterized for this combo. Runnable from anywhere -- it cd's to its
# own directory (see below).
set -euo pipefail

# --- PATH: `accelerate` lives in ~/.local/bin, which plain non-login SSH/tmux
# shells don't put on PATH (that only happens via .profile in a login shell).
# The launches/* wrapper scripts mask a missing-command failure anyway (their
# last line is `unset TOKENIZERS_PARALLELISM`, always exit 0), so this must
# be fixed here rather than relying on set -e to catch it downstream.
export PATH="$HOME/.local/bin:$PATH"
command -v accelerate >/dev/null || { echo "FATAL: accelerate not on PATH"; exit 1; }

# Checkout-relative, not a hardcoded path: these scripts used to `cd
# ~/xcube/scripts`, which broke after the repo moved to
# github.com/deb-official/plant (cloned as ~/plant) -- on a fresh GPU box
# ~/xcube does not exist, so `set -e` aborted here, and on the old dev
# machine it silently ran the FROZEN xcube checkout instead of this one.
# Resolving from BASH_SOURCE means the script always operates on the
# checkout it is part of, whatever that checkout is named or located.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Auth: scripts do hf_token = os.environ["HF_TOKEN"] (hard requirement,
# no fallback to the cached token file) -- pull it from the cache so we
# don't hardcode the secret in this script.
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export HF_USERNAME="deb101"

DATASET_BASE="wiki10-31k"
SOURCE_URL="XURLs.WIKI10_31K"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
RARE_THRESHOLD=3   # Wiki10-31K full-corpus median label frequency

source setup_plant_names.sh "${DATASET_BASE}_full" "$MODEL"
# -> DATASET_NAME=wiki10-31k_full, DATASET_SHORT=wiki10-31k, BASE_DIR=../tmp (default)

# --- Safety net: launches/* scripts don't reliably propagate accelerate's
# exit code (see PATH comment above), so verify real output landed after
# each stage instead of trusting `set -e` alone.
DATA_SUBDIR="${SOURCE_URL#XURLs.}"
verify_stage_output() {
    local desc="$1" dir="$BASE_DIR/$DATA_SUBDIR/$2"
    if [ ! -d "$dir" ] || [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "FATAL: $desc produced no output at $dir -- aborting."
        exit 1
    fi
    echo "OK: $desc -> $dir ($(ls -A "$dir" | wc -l) files)"
}

echo "=========================================="
echo "Stage 1: SOTA LLM Fine-tuning"
echo "=========================================="
launches/launchLLM \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE1_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=1 \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1
verify_stage_output "Stage 1" "$STAGE1_OUTPUT_DIR"

echo "=========================================="
echo "Stage 2: Task-Specific Head (baseline) -- also writes labels_partition.json"
echo "=========================================="
launches/launchmultilabelclas \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE2_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" --do_fresh_training \
    --num_train_epochs=3 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold="$RARE_THRESHOLD" \
    --label_coverage_frac=1.0
verify_stage_output "Stage 2" "$STAGE2_OUTPUT_DIR"

echo "=========================================="
echo "Stage 3: L2R Pretraining with MIG (needs Stage 2's labels_partition.json)"
echo "=========================================="
launches/launch_l2rboot \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" \
    --base_dir="$BASE_DIR" --output_dir="$STAGE3_OUTPUT_DIR" \
    --hub_model_id="$STAGE3_HUB_ID" --bs=8 --chnk_sz=200 --max_length=1024
verify_stage_output "Stage 3" "$STAGE3_OUTPUT_DIR"

echo "=========================================="
echo "Stage 4: L2R Attention Transfer"
echo "=========================================="
launches/launch_l2r_training \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" \
    --data_l2r_fname_prefix="$DATASET_SHORT" --max_length=512 \
    --base_dir="$BASE_DIR" --l2r_boot_dir="$STAGE3_OUTPUT_DIR" \
    --output_dir="$STAGE4_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" --model_name="$BASE_MODEL_NAME" \
    --do_fresh_training --num_train_epochs=3 --learning_rate=1e-4 --warmup_steps=0 \
    --metric_for_best_model="ndcg@25" \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1
verify_stage_output "Stage 4" "$STAGE4_OUTPUT_DIR"

echo "=========================================="
echo "Stage 5: PLANT Fine-tuning (hub_model_id = Stage 4's pushed L2R model)"
echo "=========================================="
launches/launchmultilabelclas \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE5_OUTPUT_DIR" \
    --hub_model_id="$STAGE4_HUB_ID_L2R" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-plantclassify" --do_fresh_training \
    --num_train_epochs=3 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold="$RARE_THRESHOLD" \
    --label_coverage_frac=1.0
verify_stage_output "Stage 5" "$STAGE5_OUTPUT_DIR"

echo "=========================================="
echo "ALL 5 STAGES DONE -- Wiki10-31K + Llama-3.1-8B-Instruct full-scale run complete"
echo "=========================================="
