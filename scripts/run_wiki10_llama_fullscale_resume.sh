#!/bin/bash
# Resume of run_wiki10_llama_fullscale.sh from Stage 2 onward.
# Stage 1 already succeeded and pushed to deb101/llama-8b-wiki10-31k-adapt
# before Stage 2 crashed on the pyarrow offset-overflow bug (now fixed in
# multilabel_classify.py). setup_plant_names.sh is deterministic given the
# same inputs, so re-sourcing it here reproduces the exact same Stage 1
# names without re-running Stage 1's training.
#
# Stages 2/4/5 also dropped from --per_device_train_batch_size=8 to 2 after
# Stage 2 hit a genuine CUDA OOM at the first real training step, once the
# dataset-construction fixes above got far enough to reach it: "Tried to
# allocate 14.26 GiB. GPU 0 has ... 94.23 GiB memory in use" out of 94.50
# GiB total. This is Wiki10-31K's 30,938-label per-label attention head at
# full-scale batch size, not a bug -- the smoke test never approached this
# since it capped the label space at 150. Applied to Stage 4 too as a
# precaution (same "attention over every label" scaling concern), even
# though only Stage 2 was directly observed hitting it.
#
# --gradient_accumulation_steps=4 (to preserve the effective batch size of
# 8) is only added for Stage 4 (learning2rank.py) -- confirmed via its
# --help that Stage 2/5 (multilabel_classify.py) don't expose this flag
# at all (an attempt to pass it there hit "error: unrecognized arguments:
# --gradient_accumulation_steps=4" and aborted instantly, a fast/cheap
# mistake compared to the earlier ones). Stage 2/5 just run at a smaller
# effective batch size of 2 instead.
set -euo pipefail

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

export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export HF_USERNAME="deb101"

DATASET_BASE="wiki10-31k"
SOURCE_URL="XURLs.WIKI10_31K"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
RARE_THRESHOLD=3

source setup_plant_names.sh "${DATASET_BASE}_full" "$MODEL"

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
echo "RESUMING at Stage 2 (Stage 1 already completed: $STAGE1_HUB_ID)"
echo "=========================================="

echo "=========================================="
echo "Stage 2: Task-Specific Head (baseline) -- also writes labels_partition.json"
echo "=========================================="
launches/launchmultilabelclas \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE2_OUTPUT_DIR" \
    --hub_model_id="$STAGE1_HUB_ID" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-classify" --do_fresh_training \
    --num_train_epochs=3 --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
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
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=4
verify_stage_output "Stage 4" "$STAGE4_OUTPUT_DIR"

echo "=========================================="
echo "Stage 5: PLANT Fine-tuning (hub_model_id = Stage 4's pushed L2R model)"
echo "=========================================="
launches/launchmultilabelclas \
    --source_url="$SOURCE_URL" --data="$DATASET_NAME" --max_length=512 \
    --base_dir="$BASE_DIR" --output_dir="$STAGE5_OUTPUT_DIR" \
    --hub_model_id="$STAGE4_HUB_ID_L2R" --model_name="$BASE_MODEL_NAME" \
    --task="multilabel-plantclassify" --do_fresh_training \
    --num_train_epochs=3 --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --learning_rate=1e-4 --final_lr_scheduling=1e-6 --warmup_steps=5 \
    --metric_for_best_model="precision_at_25" --rare_threshold="$RARE_THRESHOLD" \
    --label_coverage_frac=1.0
verify_stage_output "Stage 5" "$STAGE5_OUTPUT_DIR"

echo "=========================================="
echo "ALL 5 STAGES DONE -- Wiki10-31K + Llama-3.1-8B-Instruct full-scale run complete"
echo "=========================================="
