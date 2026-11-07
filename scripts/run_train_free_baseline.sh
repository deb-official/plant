#!/bin/bash
# Reproduce the training-free LLM baseline for MIMIC-IV-few (README_PLANT_FEW_SHOT.md,
# "Training-free LLM baseline"). Runs both prompting regimes (zero / few in-context
# examples) for the two paper models, on the SAME 167 held-out notes and the SAME
# bucketed precision@k as Stage 2 / Stage 5. Off-the-shelf Instruct checkpoints
# (no MIMIC training). ~30-45 min per (model,mode) on one 80GB+ GPU.
#
# Prereqs: the PLANT pipeline must have produced labels_partition.json (bucket
# defs) and create_mimic4_few.py the mimic4_icd10_few.csv. Llama needs HF gating.
# jinja2>=3.1 is required for chat templates (`pip install -U jinja2`).
set -euo pipefail
cd "$(dirname "$0")"
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export TOKENIZERS_PARALLELISM=False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CACHE="${1:-./train_free_cache}"

for MODEL in "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Llama-3.1-8B-Instruct"; do
  for MODE in zero few; do
    echo "########## train-free :: $MODEL :: $MODE :: $(date +%H:%M:%S) ##########"
    python3 train_free_baseline.py --model "$MODEL" --mode "$MODE" \
      --do all --cache_dir "$CACHE"
  done
done
echo "########## train-free :: ALL DONE :: $(date +%H:%M:%S) ##########"
