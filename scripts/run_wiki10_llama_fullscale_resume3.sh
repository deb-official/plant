#!/bin/bash
# Resume of run_wiki10_llama_fullscale_resume2.sh from Stage 4 onward.
# Stage 3 (L2R Pretraining with MIG) succeeded and is verified (8 files,
# including the two ~30-35GB rank tables) at
# ../tmp/WIKI10_31K/wiki10-31k_l2rboot_llama8b. Stage 4 crashed with a CUDA
# OOM (scored_tokens/lookup_table built directly on GPU in preprocess_data,
# ~63GB + ~15-30GB on top of the already-loaded quantized+LoRA model) --
# now fixed in scripts/learning2rank.py and xcube/l2r/models/core.py (both
# kept CPU-resident; only small per-batch slices move to GPU). Also fixed an
# unrelated bug where learning2rank.py's training entry point always
# exited 0 regardless of success/failure (a stray `finally: sys.exit(0)`
# masked the OOM at the process level). setup_plant_names.sh is
# deterministic given the same inputs, so re-sourcing it here reproduces the
# exact same Stage 1/2/3 names without re-running that work.
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
    local desc="$1" dir="$BASE_DIR/$DATA_SUBDIR/$2" min_files="${3:-1}"
    if [ ! -d "$dir" ]; then
        echo "FATAL: $desc produced no output at $dir -- aborting."
        exit 1
    fi
    local n
    n=$(ls -A "$dir" 2>/dev/null | wc -l)
    if [ "$n" -lt "$min_files" ]; then
        echo "FATAL: $desc produced only $n file(s) at $dir, expected at least $min_files -- aborting."
        exit 1
    fi
    echo "OK: $desc -> $dir ($n files)"
}

echo "=========================================="
echo "RESUMING at Stage 4 (Stage 1-3 already done and verified)"
echo "=========================================="
verify_stage_output "Stage 3 (pre-check)" "$STAGE3_OUTPUT_DIR" 8

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
verify_stage_output "Stage 4" "$STAGE4_OUTPUT_DIR" 10

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
verify_stage_output "Stage 5" "$STAGE5_OUTPUT_DIR" 10

echo "=========================================="
echo "ALL 5 STAGES DONE -- Wiki10-31K + Llama-3.1-8B-Instruct full-scale run complete"
echo "=========================================="
