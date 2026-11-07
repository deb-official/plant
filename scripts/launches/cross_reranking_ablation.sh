#!/bin/bash

# Check if at least one predict_topk value is provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 --predict_topk <value1> <value2> ... <valueN>"
  exit 1
fi

# Parse the arguments after `--predict_topk`
if [ "$1" == "--predict_topk" ]; then
  shift # Shift past the `--predict_topk` argument
  predict_topk_list=("$@")
else
  echo "Error: Missing --predict_topk argument."
  exit 1
fi

# Iterate over each predict_topk value and run the command
for predict_topk in "${predict_topk_list[@]}"; do
  echo "Running with predict_topk: $predict_topk"

  launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --min_cluster_size=15 \
    --predict \
    --predict_topk="${predict_topk}" \
    --cross_rerank \
    --logfile="cross_reranking_${predict_topk}"
done