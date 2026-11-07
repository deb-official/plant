#!/bin/bash

# Check if at least one gpl_steps value is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 --gpl_steps <value1> <value2> ... <valueN>"
  exit 1
fi

# Parse the arguments after `--gpl_steps`
if [ "$1" == "--gpl_steps" ]; then
  shift # Shift past the `--gpl_steps` argument
  gpl_steps_list=("$@")
else
  echo "Error: Missing --gpl_steps argument."
  exit 1
fi

# Iterate over each gpl_steps value and run the command
for gpl_steps in "${gpl_steps_list[@]}"; do
  echo "Running with GPL steps: $gpl_steps"

  launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --train_margin_mseloss \
    --negatives_per_query=50 \
    --gpl_steps="${gpl_steps}" \
    --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
    --retrievers="['msmarco-distilbert-base-v3', 'hkunlp/instructor-large']" \
    --do_evaluation \
    --logfile="margin_mseloss_traineval_${gpl_steps}"
done