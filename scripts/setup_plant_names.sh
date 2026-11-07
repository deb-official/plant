#!/bin/bash
# setup_plant_names.sh - Generate consistent names for PLANT pipeline

# Configuration
HF_USERNAME=${HF_USERNAME:-"deb101"}  # Set your Hugging Face username
DATASET=${1:-"mimic4_icd10_full"}
BASE_MODEL=${2:-"mistralai/Mistral-7B-Instruct-v0.3"}
BASE_DIR=${3:-"../tmp"}

# Extract model info
MODEL_PROVIDER=$(echo $BASE_MODEL | cut -d'/' -f1)
MODEL_FULL_NAME=$(echo $BASE_MODEL | cut -d'/' -f2)
# `|| true`: under `set -e`/`pipefail` (e.g. a caller wrapping this script in
# a strict shell block), grep finding no match -- e.g. Phi-3-small-8k-instruct,
# whose name has no [0-9]+[BMG] token, just "8k" -- would otherwise abort the
# whole script here before the microsoft-provider branch below even runs and
# correctly overrides MODEL_SIZE/MODEL_SHORT for it.
MODEL_SIZE=$(echo $MODEL_FULL_NAME | grep -oE '[0-9]+[BMG]' | head -1) || true
MODEL_TYPE=$(echo $MODEL_FULL_NAME | cut -d'-' -f1 | tr '[:upper:]' '[:lower:]')

# Generate short names
if [[ $MODEL_PROVIDER == "mistralai" ]]; then
    MODEL_SHORT="mistral${MODEL_SIZE,,}"
elif [[ $MODEL_PROVIDER == "meta-llama" ]]; then
    MODEL_SHORT="llama${MODEL_SIZE,,}"
elif [[ $MODEL_PROVIDER == "deepseek-ai" ]]; then
    MODEL_SHORT="deepseek${MODEL_SIZE,,}"
elif [[ $MODEL_PROVIDER == "microsoft" ]]; then
    # Phi-3 repo names encode context length (e.g. "4k"/"8k"), not a
    # [0-9]+[BMG] parameter count, so the shared MODEL_SIZE regex above
    # finds nothing -- every Phi-3 variant would otherwise collapse to the
    # same short name "phi" and collide on output dirs/Hub ids. Derive the
    # short name from the mini/small/medium variant word instead.
    MODEL_TYPE="phi3"
    MODEL_SIZE=$(echo $MODEL_FULL_NAME | cut -d'-' -f3 | tr '[:upper:]' '[:lower:]')
    MODEL_SHORT="phi3${MODEL_SIZE}"
else
    MODEL_SHORT="${MODEL_TYPE}${MODEL_SIZE,,}"
fi

# Dataset short name: strip the trailing "_full"-style suffix, the same way
# bootl2r.py derives its file-naming prefix ('_'.join(data.split('_')[:-1])).
# A fixed "-f1-2" cut breaks for single-segment/hyphenated names like
# "amazon-3m_full" or "amazon-670k_full" (only 2 underscore fields, so
# "-f1-2" returns the whole string, suffix included).
DATASET_SHORT="${DATASET%_*}"

# Export names for each stage
export BASE_MODEL_NAME="$BASE_MODEL"
export DATASET_NAME="$DATASET"
export BASE_DIR="$BASE_DIR"

# Stage 1: SOTA LLM Fine-tuning
export STAGE1_HUB_ID="${HF_USERNAME}/${MODEL_TYPE}-${MODEL_SIZE,,}-${DATASET_SHORT}-adapt"
export STAGE1_OUTPUT_DIR="${MODEL_SHORT}_${DATASET_SHORT}"

# Stage 2: Task-Specific Head Attachment (for baseline comparison only)
export STAGE2_HUB_ID="$STAGE1_HUB_ID"
export STAGE2_OUTPUT_DIR="${DATASET_SHORT}_classify_${MODEL_SHORT}"

# Stage 3: L2R Pretraining with MIG
export STAGE3_HUB_ID="$BASE_MODEL_NAME"  # Uses base model tokenizer
export STAGE3_OUTPUT_DIR="${DATASET_SHORT}_l2rboot_${MODEL_SHORT}"

# Stage 4: L2R Attention Transfer
export STAGE4_HUB_ID="$STAGE1_HUB_ID"
export STAGE4_OUTPUT_DIR="${DATASET_SHORT}_l2rtrain_${MODEL_SHORT}"
export STAGE4_HUB_ID_L2R="${STAGE1_HUB_ID}-l2r"  # Output model with L2R

# Stage 5: PLANT Fine-tuning
export STAGE5_HUB_ID="$STAGE4_HUB_ID_L2R"
export STAGE5_OUTPUT_DIR="${DATASET_SHORT}_plantclassify_${MODEL_SHORT}"

# Print configuration
echo "🌱 PLANT Pipeline Configuration"
echo "================================"
echo "Base Model: $BASE_MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Model Short Name: $MODEL_SHORT"
echo ""
echo "📋 Stage Names:"
echo "Stage 1 - SOTA LLM Fine-tuning:"
echo "  Hub ID: $STAGE1_HUB_ID"
echo "  Output: $STAGE1_OUTPUT_DIR"
echo ""
echo "Stage 2 - Task-Specific Head Attachment (for baseline comparison only):"
echo "  Hub ID: $STAGE2_HUB_ID"
echo "  Output: $STAGE2_OUTPUT_DIR"
echo ""
echo "Stage 3 - L2R Pretraining with MIG:"
echo "  Hub ID: $STAGE3_HUB_ID"
echo "  Output: $STAGE3_OUTPUT_DIR"
echo ""
echo "Stage 4 - L2R Attention Transfer:"
echo "  Hub ID: $STAGE4_HUB_ID"
echo "  Output: $STAGE4_OUTPUT_DIR"
echo "  L2R Model: $STAGE4_HUB_ID_L2R"
echo ""
echo "Stage 5 - PLANT Fine-tuning:"
echo "  Hub ID: $STAGE5_HUB_ID"
echo "  Output: $STAGE5_OUTPUT_DIR"
echo ""
echo "✅ Environment variables exported!"