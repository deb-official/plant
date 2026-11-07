#!/bin/bash

# Initialize empty arrays for min_topicsz and min_cluster_size
min_topicsz_list=()
min_cluster_list=()

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --min_topicsz) shift; min_topicsz_list=("$@"); break ;;
    --min_cluster_size) shift; min_cluster_list=("$@"); break ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Ensure at least one of the two options is provided
if [[ -z "${min_topicsz_list[*]}" && -z "${min_cluster_list[*]}" ]]; then
  echo "Error: You must provide at least one of --min_topicsz or --min_cluster_size."
  echo "Usage: $0 [--min_topicsz <values...>] [--min_cluster_size <values...>]"
  exit 1
fi

# Run commands for min_topicsz if provided
if [[ -n "${min_topicsz_list[*]}" ]]; then
  for min_topicsz in "${min_topicsz_list[@]}"; do
    echo "Processing for min_topicsz: $min_topicsz"
    base_dir="../tmp2"
    logfile_train="topic_model_train_${min_topicsz}"
    logfile_pseudolabel="topic_model_pseudolabel_${min_topicsz}"
    logfile_predict="topic_model_predict_${min_topicsz}"
    embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b"
    source_url="XURLs.LF_WIKISEEALSO_320K"
    dataset="deb101/lf_wikiseealso_320k"
    train_test_file_path="lf-wikiseealso-320k.csv" 

    # Command 1: Train
    launches/topic_modelling_zeroshot.sh \
      --embedding_model_name="${embedding_model_name}" \
      --base_dir="${base_dir}" \
      --source_url="${source_url}" \
      --dataset="${dataset}" \
      --min_topic_size="${min_topicsz}" \
      --logfile="${logfile_train}"

    # Command 2: Pseudolabel
    launches/topic_modelling_zeroshot.sh \
      --embedding_model_name="${embedding_model_name}" \
      --base_dir="${base_dir}" \
      --source_url="${source_url}" \
      --dataset="${dataset}" \
      --min_topic_size="${min_topicsz}" \
      --pseudolabel \
      --logfile="${logfile_pseudolabel}"

    # Command 3: Predict
    launches/topic_modelling_zeroshot.sh \
      --embedding_model_name="${embedding_model_name}" \
      --base_dir="${base_dir}" \
      --source_url="${source_url}" \
      --dataset="${dataset}" \
      --min_topic_size="${min_topicsz}" \
      --train_test_file_path="${train_test_file_path}"\
      --predict \
      --logfile="${logfile_predict}"
  done
fi

# Run commands for min_cluster_size if provided
if [[ -n "${min_cluster_list[*]}" ]]; then
  for min_cluster_size in "${min_cluster_list[@]}"; do
    echo "Processing for min_cluster_size: $min_cluster_size"
    base_dir="../tmp2"
    logfile_train="topic_model_train_${min_cluster_size}"
    logfile_pseudolabel="topic_model_pseudolabel_${min_cluster_size}"
    logfile_predict="topic_model_predict_${min_cluster_size}"
    embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b"
    source_url="XURLs.LF_WIKISEEALSO_320K"
    dataset="deb101/lf_wikiseealso_320k"
    train_test_file_path="lf-wikiseealso-320k.csv" 

    # Command 1: Train
    launches/topic_modelling_zeroshot.sh \
      --embedding_model_name="${embedding_model_name}" \
      --base_dir="${base_dir}" \
      --source_url="${source_url}" \
      --dataset="${dataset}" \
      --min_cluster_size="${min_cluster_size}" \
      --logfile="${logfile_train}" \
      # --no_fresh_topicmodel # add this if you do not want to train a fresh topic model

    # Command 2: Pseudolabel
    launches/topic_modelling_zeroshot.sh \
      --embedding_model_name="${embedding_model_name}" \
      --base_dir="${base_dir}" \
      --source_url="${source_url}" \
      --dataset="${dataset}" \
      --min_cluster_size="${min_cluster_size}" \
      --pseudolabel \
      --logfile="${logfile_pseudolabel}"

    # Command 3: Predict
    launches/topic_modelling_zeroshot.sh \
      --embedding_model_name="${embedding_model_name}" \
      --base_dir="${base_dir}" \
      --source_url="${source_url}" \
      --dataset="${dataset}" \
      --min_cluster_size="${min_cluster_size}" \
      --train_test_file_path="${train_test_file_path}"\
      --predict \
      --logfile="${logfile_predict}"
  done
fi