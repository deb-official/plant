#!/bin/bash

# Script: fewshots.sh
# Usage: ./fewshots.sh --label_frac_sup 0 1 2 3 6 10

# Function to print usage instructions
print_usage() {
    echo "Usage: $0 --label_frac_sup <list of label fractions> --gpl_steps <list of gpl_steps>"
    echo "Example: $0 --label_frac_sup 0 1 2 3 6 10 --gpl_steps 50 100 200 400 600 800"
    exit 1
}

# Check if parameters are provided
if [[ "$#" -lt 4 ]]; then
    echo "Error: Missing required parameters."
    print_usage
fi

# Parse arguments
label_frac_sup=()
gpl_steps=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --label_frac_sup)
            shift
            while [[ "$#" -gt 0 && $1 != --* ]]; do
                label_frac_sup+=("$1")
                shift
            done
            ;;
        --gpl_steps)
            shift
            while [[ "$#" -gt 0 && $1 != --* ]]; do
                gpl_steps+=("$1")
                shift
            done
            ;;
        *)
            echo "Error: Unknown parameter passed: $1"
            print_usage
            ;;
    esac
done

# Print parsed arguments
echo "Parsed Arguments:"
echo "label_frac_sup: ${label_frac_sup[*]}"
echo "gpl_steps: ${gpl_steps[*]}"

# Wait for user confirmation to proceed
echo "Please confirm the parsed arguments. Press Enter to continue or Ctrl+C to abort..."
read -r  # Waits for the user to press Enter

# Ensure both lists are provided and have the same size
if [[ ${#label_frac_sup[@]} -eq 0 || ${#gpl_steps[@]} -eq 0 ]]; then
    echo "Error: Both --label_frac_sup and --gpl_steps must be provided."
    print_usage
fi

if [[ ${#label_frac_sup[@]} -ne ${#gpl_steps[@]} ]]; then
    echo "Error: The size of --label_frac_sup and --gpl_steps lists must match."
    print_usage
fi

# Initialize dataset and base directory
dataset="LF_AMAZON_131K_sample"
base_dir_linux="../tmp_ablation/min_cluster_15/$dataset"
base_dir="../tmp_ablation/min_cluster_15"

# Print the base directory
echo "Base directory set to: $base_dir"

# Iterate over each pair of label_frac_sup and gpl_steps values
for i in "${!label_frac_sup[@]}"; do
    label="${label_frac_sup[$i]}"
    gpl_step="${gpl_steps[$i]}"
    echo "********** STARTING PROCESSING FOR label_frac_sup=$label and gpl_steps=$gpl_step **********"

    # Step 1: Remove specified files and directories
    echo "Removing files and directories for label_frac_sup=$label..."
    rm -f "$base_dir_linux/hard-negatives.jsonl" && \
    rm -f "$base_dir_linux/gpl-training-data.tsv" && \
    
    if [[ $? -eq 0 ]]; then
        echo "All files and directories successfully removed for label_frac_sup=$label."
    else
        echo "Error removing files or directories for label_frac_sup=$label."
    fi

    # Step 2: Run the first command
    echo "Running topic_modelling_zeroshot.sh for data preparation..."
    launches/topic_modelling_zeroshot.sh \
        --embedding_model_name=hkunlp/instructor-large \
        --base_dir="$base_dir" \
        --source_url=XURLs.LF_AMAZON_131K_sample \
        --dataset=deb101/lf_amazon_131k_sample \
        --num_pos=10 \
        --prep_margin_mseloss_data \
        --label_frac_sup="$label" \
        --logfile=fewshot_margin_mseloss_dataprep_with_label_sup_"$label"

    # Step 3: Run the second command
    echo "Running topic_modelling_zeroshot.sh for training and evaluation..."
    launches/topic_modelling_zeroshot.sh \
        --embedding_model_name=hkunlp/instructor-large \
        --base_dir="$base_dir" \
        --source_url=XURLs.LF_AMAZON_131K_sample \
        --dataset=deb101/lf_amazon_131k_sample \
        --train_margin_mseloss \
        --negatives_per_query=50 \
        --gpl_steps="$gpl_step" \
        --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
        --retrievers="['msmarco-distilbert-base-v3', 'hkunlp/instructor-large']" \
        --do_evaluation \
        --logfile=fewshot_margin_mseloss_train_eval_"$label"_gpl"$gpl_step"

    echo "********** COMPLETED PROCESSING FOR label_frac_sup=$label and gpl_steps=$gpl_step **********"

done