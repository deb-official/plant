#!/bin/bash
# =============================================================================
# Launch Script for Learning-to-Rank Training
# =============================================================================
#
# This script configures and launches a Learning-to-Rank (L2R) training process
# for a language model, specifically focused on medical datasets (MIMIC4).
# It provides a standardized way to handle command-line arguments, validate inputs,
# and execute training with proper logging and GPU acceleration.
#
# === HOW TO USE THIS SCRIPT ===
# 1. Basic usage with default parameters:
#    ./launch_l2r_training.sh
#
# 2. Custom training run:
#    ./launch_l2r_training.sh \
#      --source_url="XURLs.MIMIC4_DEMO" \
#      --data="mimic4_icd10_full" \
#      --data_l2r="mimic4_icd10_tok_lbl" \
#      --output_dir="custom_output" \
#      --do_fresh_training \
#      --num_train_epochs=10
#
# For more detailed help, run this script with the --extended_help option.
# =============================================================================

# --------------------------------
# Default Configuration Parameters
# --------------------------------
# Data parameters
SOURCE_URL="XURLs.MIMIC4_DEMO"
DATA="mimic4_icd10_full"
DATA_L2R="mimic4_icd10_tok_lbl"
MAX_LENGTH=512

# Directory parameters
BASE_DIR="../tmp"
L2R_BOOT_DIR="mimic4_l2rboot"
OUTPUT_DIR="mistral_mimic4_l2r"

# Model parameters
HUB_MODEL_ID="deb101/mistral-7b-instruct-v0.3-mimic4-adapt"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"

# Training parameters
DO_FRESH_TRAINING=false
NUM_TRAIN_EPOCHS=5
LEARNING_RATE=1e-4
WARMUP_STEPS=500
VERBOSE=false

# GPU configuration
GPU_IDS="0"
PRECISION="fp16"  # Options: fp16, bf16, fp32

# Array to store additional unknown arguments
EXTRA_ARGS=()

# -------------------
# Extended Help Function
# -------------------
function show_extended_help {
    echo "==================================================================="
    echo "           LEARNING-TO-RANK (L2R) TRAINING LAUNCH SCRIPT           "
    echo "==================================================================="
    echo
    echo "This script launches a Learning-to-Rank training process for"
    echo "language models, specifically designed for medical datasets (MIMIC4)."
    echo "It uses the HuggingFace's Accelerate library for distributed training."
    echo
    echo "=== SCRIPT FUNCTIONALITY ==="
    echo
    echo "1. PARAMETER HANDLING"
    echo "   The script accepts various command-line arguments to customize"
    echo "   the training process, validates them, and passes them to the"
    echo "   Python training script."
    echo
    echo "2. ENVIRONMENT SETUP"
    echo "   It sets necessary environment variables, activates virtual"
    echo "   environments if available, and creates required directories."
    echo
    echo "3. GPU ACCELERATION"
    echo "   The script uses HuggingFace's Accelerate library to optimize"
    echo "   training on GPUs with mixed precision options."
    echo
    echo "4. LOGGING"
    echo "   All training output is logged to a timestamped file for future reference."
    echo
    echo "=== EXAMPLE USAGE ==="
    echo
    echo "Basic usage with default parameters:"
    echo "  ./launch_l2r_training.sh"
    echo
    echo "Custom training run:"
    echo "  ./launch_l2r_training.sh --source_url=\"XURLs.MIMIC4_DEMO\" --data=\"mimic4_icd10_full\" --data_l2r=\"mimic4_icd10_tok_lbl\" --output_dir=\"custom_output\" --do_fresh_training --num_train_epochs=10 --precision=\"bf16\""
    echo
    echo "For quick parameter reference, use --help instead of --extended_help"
    echo "==================================================================="
    exit 0
}

# -------------------
# Basic Help Function
# -------------------
function show_help {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    # Data parameters
    echo "  --source_url=URL       Source URL for data (default: $SOURCE_URL)"
    echo "  --data=NAME            Dataset name (default: $DATA)"
    echo "  --data_l2r=NAME        Tokenized and labeled dataset for L2R (default: $DATA_L2R)"
    echo "  --max_length=NUM       Maximum sequence length (default: $MAX_LENGTH)"
    # Directory parameters
    echo "  --base_dir=DIR         Base directory for temporary files (default: $BASE_DIR)"
    echo "  --l2r_boot_dir=DIR     Directory for bootstrap data for L2R (default: $L2R_BOOT_DIR)"
    echo "  --output_dir=DIR       Output directory for model and logs (default: $OUTPUT_DIR)"
    # Model parameters
    echo "  --hub_model_id=ID      Hugging Face model ID (default: $HUB_MODEL_ID)"
    echo "  --model_name=NAME      Base model name (default: $MODEL_NAME)"
    # Training parameters
    echo "  --do_fresh_training    Enable fresh training (default: disabled)"
    echo "  --num_train_epochs=NUM Number of training epochs (default: $NUM_TRAIN_EPOCHS)"
    echo "  --learning_rate=NUM    Learning rate (default: $LEARNING_RATE)"
    echo "  --warmup_steps=NUM     Warmup steps (default: $WARMUP_STEPS)"
    # GPU configuration
    echo "  --gpu_ids=IDS          Comma-separated GPU IDs (default: $GPU_IDS)"
    echo "  --precision=TYPE       Training precision: fp16, bf16, fp32 (default: $PRECISION)"
    # Misc
    echo "  --verbose              Enable verbose output"
    echo "  --help                 Show this help message"
    echo "  --extended_help        Show detailed information about using this script"
    echo "  [additional options]   Any other arguments are passed to the Python script"
    echo
    echo "Example:"
    echo "  $0 --source_url=\"XURLs.MIMIC4_DEMO\" --data=\"mimic4_icd10_full\" --data_l2r=\"mimic4_icd10_tok_lbl\" --base_dir=\"../tmp\" --l2r_boot_dir=\"mimic4_l2rboot\" --output_dir=\"results\" --do_fresh_training --verbose"
    echo
    echo "For more detailed help and information about this script, use --extended_help"
    echo
}

# -------------------
# Parse Arguments
# -------------------
while [ $# -gt 0 ]; do
    case "$1" in
        # Help options
        --help)
            show_help
            exit 0
            ;;
        --extended_help)
            show_extended_help
            exit 0
            ;;
        # Data parameters
        --source_url=*)
            SOURCE_URL="${1#*=}"
            shift
            ;;
        --data=*)
            DATA="${1#*=}"
            shift
            ;;
        --data_l2r=*)
            DATA_L2R="${1#*=}"
            shift
            ;;
        --max_length=*)
            MAX_LENGTH="${1#*=}"
            shift
            ;;
        # Directory parameters
        --base_dir=*)
            BASE_DIR="${1#*=}"
            shift
            ;;
        --l2r_boot_dir=*)
            L2R_BOOT_DIR="${1#*=}"
            shift
            ;;
        --output_dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        # Model parameters
        --hub_model_id=*)
            HUB_MODEL_ID="${1#*=}"
            shift
            ;;
        --model_name=*)
            MODEL_NAME="${1#*=}"
            shift
            ;;
        # Training parameters
        --do_fresh_training)
            DO_FRESH_TRAINING=true
            shift
            ;;
        --num_train_epochs=*)
            NUM_TRAIN_EPOCHS="${1#*=}"
            shift
            ;;
        --learning_rate=*)
            LEARNING_RATE="${1#*=}"
            shift
            ;;
        --warmup_steps=*)
            WARMUP_STEPS="${1#*=}"
            shift
            ;;
        # GPU configuration
        --gpu_ids=*)
            GPU_IDS="${1#*=}"
            shift
            ;;
        --precision=*)
            PRECISION="${1#*=}"
            shift
            ;;
        # Misc
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            # Store unknown arguments
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# -------------------
# Validate Inputs
# -------------------
# Required parameters check
if [ -z "$SOURCE_URL" ] || [ -z "$DATA" ] || [ -z "$DATA_L2R" ] || [ -z "$BASE_DIR" ] || [ -z "$L2R_BOOT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --source_url, --data, --data_l2r, --base_dir, --l2r_boot_dir, and --output_dir cannot be empty"
    show_help
    exit 1
fi

# Validate numeric parameters
if ! [[ "$MAX_LENGTH" =~ ^[0-9]+$ ]] || ! [[ "$NUM_TRAIN_EPOCHS" =~ ^[0-9]+$ ]] || ! [[ "$WARMUP_STEPS" =~ ^[0-9]+$ ]]; then
    echo "Error: --max_length, --num_train_epochs, and --warmup_steps must be positive integers"
    show_help
    exit 1
fi

if ! [[ "$LEARNING_RATE" =~ ^[0-9]*\.?[0-9]+(e-[0-9]+)?$ ]]; then
    echo "Error: --learning_rate must be a valid float or scientific notation (e.g., 1e-4)"
    show_help
    exit 1
fi

# Validate precision
if [[ "$PRECISION" != "fp16" && "$PRECISION" != "bf16" && "$PRECISION" != "fp32" ]]; then
    echo "Error: --precision must be one of: fp16, bf16, fp32"
    show_help
    exit 1
fi

# -------------------
# Environment Setup
# -------------------
# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Set the main training script path
TRAINING_SCRIPT="$PROJECT_ROOT/learning2rank.py"

# Check if the script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "Error: Python script not found at $TRAINING_SCRIPT"
    echo "Make sure the learning2rank.py file is in the project root directory."
    exit 1
fi

# Activate virtual environment if it exists
VENV_PATH="$PROJECT_ROOT/venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

# Create output timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_${TIMESTAMP}.log"

# Create required directories
mkdir -p "$BASE_DIR"
mkdir -p "$OUTPUT_DIR"

# -------------------
# Display Configuration
# -------------------
echo "Starting L2R training at $(date)..."
echo "================================================================================"
echo "LAUNCHING LEARNING-TO-RANK TRAINING"
echo "================================================================================"
# Data parameters
echo "Source URL: $SOURCE_URL"
echo "Dataset: $DATA"
echo "L2R Dataset: $DATA_L2R"
echo "Max Length: $MAX_LENGTH"

# Directory parameters
echo "Base Directory: $BASE_DIR"
echo "L2R Bootstrap Directory: $L2R_BOOT_DIR"
echo "Output Directory: $OUTPUT_DIR"

# Model parameters
echo "Hub Model ID: $HUB_MODEL_ID"
echo "Model Name: $MODEL_NAME"

# Training parameters
echo "Fresh Training: $DO_FRESH_TRAINING"
echo "Number of Epochs: $NUM_TRAIN_EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Warmup Steps: $WARMUP_STEPS"

# GPU configuration
echo "GPU IDs: $GPU_IDS"
echo "Precision: $PRECISION"

# Extra parameters
echo "Verbose: $VERBOSE"
echo "Additional Arguments: ${EXTRA_ARGS[*]}"
echo "Log File: $OUTPUT_DIR/$LOG_FILE"
echo "--------------------------------------------------------------------------------"

# -------------------
# Prepare Arguments
# -------------------
# Build the command arguments
ARGS=(
    # Data parameters
    --source_url="$SOURCE_URL"
    --data="$DATA"
    --data_l2r="$DATA_L2R"
    --max_length="$MAX_LENGTH"
    
    # Directory parameters
    --base_dir="$BASE_DIR"
    --l2r_boot_dir="$L2R_BOOT_DIR"
    --output_dir="$OUTPUT_DIR"
    
    # Model parameters
    --hub_model_id="$HUB_MODEL_ID"
    --model_name="$MODEL_NAME"
    
    # Training parameters
    --num_train_epochs="$NUM_TRAIN_EPOCHS"
    --learning_rate="$LEARNING_RATE"
    --warmup_steps="$WARMUP_STEPS"
)

# Add boolean flags
if [ "$DO_FRESH_TRAINING" = true ]; then
    ARGS+=(--do_fresh_training)
fi

if [ "$VERBOSE" = true ]; then
    ARGS+=(--verbose)
fi

# Append extra arguments
ARGS+=("${EXTRA_ARGS[@]}")

# -------------------
# Execute Training
# -------------------
# Set environment variables
export TOKENIZERS_PARALLELISM=False
# Add any other environment variables needed here

# Execute with accelerate
accelerate launch --mixed_precision="$PRECISION" --gpu_ids "$GPU_IDS" \
    "$TRAINING_SCRIPT" "${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/$LOG_FILE"

# Clean up environment variables
unset TOKENIZERS_PARALLELISM
# Unset other environment variables here

# -------------------
# Check Results
# -------------------
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully at $(date)."
else
    echo "Error executing training. Exit code: $EXIT_CODE. See $OUTPUT_DIR/$LOG_FILE for details."
    exit $EXIT_CODE
fi