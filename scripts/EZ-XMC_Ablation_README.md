# Ablation Analysis in Unsupervised Zero-Shot Setting

## 1. Influence of Topic Modelling

The following will generate a sample of the dataset
```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$ ./launches/generate_ablat.sh
```
Tar the data that was generated at the source and copy to aws:
```sh
(base) deb@deb-GIGABYTE:~/.xcube/data/LF-Amazon-131K_sample$ tree
.
├── lbl.json
├── lf-amazon-131k.csv
├── trn.json
└── tst.json

0 directories, 4 files
cd ..
tar -czf LF-Amazon-131K_sample.tar.gz LF-Amazon-131K_sample/
aws s3 cp LF-Amazon-131K_sample.tar.gz s3://xcubebucket/lf-amazon-131k/
```

Also, upload the above generated files to huggingface dataset. Make sure to update the README.md reflecting the split information. 

### 1.1 Minimum Cluster Size:

1. Train the topic model and toicfy the data:
(Uncomment the openai_api key in topic_modelling_zeroshot.sh)
```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --min_cluster_size=15 \
    --logfile=topic_model_train
```

2. Pseudo-Label the data: By adding the flag `--pseudolabel`
```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --min_cluster_size=15 \
    --pseudolabel \
    --logfile=topic_model_pseudolabel
```

3. Make predictions on test set: By adding the flag `predict`
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --min_cluster_size=15 \
    --predict \
    --logfile=topic_model_predict
```

Now to make sure we can use the train_doc, test_doc and label embeddings:
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation$
for dir in min_cluster_{40,50,60,70}; do
    mkdir -p "$dir/LF_AMAZON_131K_sample"
    ln -s "$(pwd)/min_cluster_15/LF_AMAZON_131K_sample/label_embeddings.pt" "$dir/LF_AMAZON_131K_sample/label_embeddings.pt"
    ln -s "$(pwd)/min_cluster_15/LF_AMAZON_131K_sample/doc_embeddings.pt" "$dir/LF_AMAZON_131K_sample/doc_embeddings.pt"
    ln -s "$(pwd)/min_cluster_15/LF_AMAZON_131K_sample/test_doc_embeddings.pt" "$dir/LF_AMAZON_131K_sample/test_doc_embeddings.pt"
done
```

Then to perform topicfying, pseudplabelling and predicting using a script:
```sh
launches/topics_ablation.sh 5 30 40 50 60 70 80 90 100 110 120 130
```

### 1.2 Minimum Topic Size

```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation$
for dir in min_topicsz_{20,40,60,80,100,120,140,160,180,200}; do
    mkdir -p "$dir/LF_AMAZON_131K_sample"
    ln -s "$(pwd)/min_cluster_15/LF_AMAZON_131K_sample/label_embeddings.pt" "$dir/LF_AMAZON_131K_sample/label_embeddings.pt"
    ln -s "$(pwd)/min_cluster_15/LF_AMAZON_131K_sample/doc_embeddings.pt" "$dir/LF_AMAZON_131K_sample/doc_embeddings.pt"
    ln -s "$(pwd)/min_cluster_15/LF_AMAZON_131K_sample/test_doc_embeddings.pt" "$dir/LF_AMAZON_131K_sample/test_doc_embeddings.pt"
done
```

```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$ launches/topics_ablation.sh --min_topicsz 20 40 60 80 100 120 140 160 180 200
```

Once the ablations are done, you can check the metrics using:
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation$ find min_topicsz_* -name "topic_model_predict_*" | sort -t'_' -k3n | xargs -I {} grep --color=always -wH "NDCG@10" {}

(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation$ find min_cluster_* -name "topic_model_predict_*" | sort -t'_' -k3n | xargs -I {} grep --color=always -wH "P@3:" {}
```

## 2. Influence of Margin-MSE Loss Training

1. Prepare the datasets in BeIR format for margin-mseloss training with:

```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --num_pos=10 \
    --prep_margin_mseloss_data \
    --logfile=margin_mseloss_dataprep
```
output:
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation$ find min_cluster_15 -mmin -30
min_cluster_15/LF_AMAZON_131K_sample
min_cluster_15/LF_AMAZON_131K_sample/corpus.jsonl
min_cluster_15/LF_AMAZON_131K_sample/qgen-queries.jsonl
min_cluster_15/LF_AMAZON_131K_sample/qgen-qrels
min_cluster_15/LF_AMAZON_131K_sample/qgen-qrels/train.tsv
min_cluster_15/LF_AMAZON_131K_sample/qrels
min_cluster_15/LF_AMAZON_131K_sample/qrels/test.tsv
min_cluster_15/LF_AMAZON_131K_sample/queries.jsonl
```

2. Now to perform the margin-mseloss training data generation with:

```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --train_margin_mseloss \
    --negatives_per_query=50 \
    --gpl_steps=140000 \
    --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
    --retrievers="['msmarco-distilbert-base-v3', 'hkunlp/instructor-large']" \
    --logfile=margin_mseloss_train_data_prep \
    --stop_after_gpl_train_data_creation
```

This will generate:
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation$ find min_cluster_15/LF_AMAZON_131K_sample/ -maxdepth 1 -mmin -30
min_cluster_15/LF_AMAZON_131K_sample/
min_cluster_15/LF_AMAZON_131K_sample/hard-negatives.jsonl # for each query there are `num_pos` many postives and `negatives_per_query` many negatives
min_cluster_15/LF_AMAZON_131K_sample/gpl-training-data.tsv
```

Note: `gpl-training-data.tsv` contains (`gpl_steps` * `batch_size_gpl`) tuples in total. This is also a quick way to find out the `gpl_steps` used to generate the `gpl-training-data.tsv`, by doing `wc -l gpl-training-data.tsv`

3. Now to perform training and evaluation just remove the flag `stop_after_gpl_train_data_creation` add the flag `do_evaluation`:

```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --train_margin_mseloss \
    --negatives_per_query=50 \
    --gpl_steps=1400 \
    --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
    --retrievers="['msmarco-distilbert-base-v3', 'hkunlp/instructor-large']" \
    --do_evaluation \
    --logfile=margin_mseloss_traineval
```

The trained model is saved at 
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation/min_cluster_15/LF_AMAZON_131K_sample$ tree -L 1 output/
output/
├── 1400
├── 1_Pooling
├── 3000
├── README.md
├── config.json
├── config_sentence_transformers.json
├── eval
├── model.safetensors
├── modules.json
├── sentence_bert_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt

4 directories, 10 files

(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation/min_cluster_15/LF_AMAZON_131K_sample$ tree -L 1 output/1400
output/1400
├── 1_Pooling
├── README.md
├── config.json
├── config_sentence_transformers.json
├── model.safetensors
├── modules.json
├── sentence_bert_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt

1 directory, 10 files
```

Note: 
1. `--retrievers` is for mining negative examples
2. You can also set `--cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2"`. This will be used in pseudo-labelling the negative examples to create `gpl-training-data.tsv`
3. `base_ckpt` is the model that gpl training finetunes with margin mseloss.

To perform train/eval on various `gpl_steps` after generating the `gpl-training-data.tsv`, we can use the following script: 

```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$ launches/margin_mseloss_ablation.sh --gpl_steps 200 414 857 1775 3677 7614 15767 32650 #67609 140000
```

After the ablations are done, we can see the results using:
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation/min_cluster_15/LF_AMAZON_131K_sample$ find logs -name "margin_mseloss_traineval_*" | sort -t'_' -k4n | xargs -I {} sh -c 'grep --color=always -wH "P@1:" {} && echo "*******"'
```

## 3. Influence of Cross-Reranking 

First, make sure we have done the stages (1) Topic Modelling and (2) of Margin-MSEloss Training:

Note: Make sure to remove
```
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation/min_cluster_15/LF_AMAZON_131K_sample$ rm tst_predicted.csv
``` 

```sh
# topic modelling
(base) deb@deb-GIGABYTE:~/xcube/scripts$
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --min_cluster_size=15 \
    --predict \
    --logfile=test_dummy

# margin-mseloss training
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --train_margin_mseloss \
    --negatives_per_query=50 \
    --gpl_steps=1400 \
    --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
    --retrievers="['msmarco-distilbert-base-v3', 'hkunlp/instructor-large']" \
    --do_evaluation \
    --logfile=test_dummy1
```
Then, we can do the reranking:
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --min_cluster_size=15 \
    --predict \
    --predict_topk=13 \
    --cross_rerank \
    --logfile="cross_reranking_13"
```
Or if you want to perform ablations:

```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$
./cross_reranking_ablation.sh --predict_topk 7 11 16 21 26 31 36 41 46 51 55 60 65 70 75 80 85 90 #95 100
```

After the ablations you can view the results as the following. This shows two lines for each log file with `predict_topk` value - first line without reranking, second line with reranking, third line is reranking and aggregated with margin-mseloss predictions:

```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation/min_cluster_15/LF_AMAZON_131K_sample$
find logs -name "cross_reranking_*" | sort -t'_' -k3n | xargs -I {} sh -c 'grep --color=always -wH "P@1:" {} && echo "*******"' 
## If you just want the second (topic+cross_reranking) line or third (topic+margin_mseloss+cross_reranking+) line 
find logs -name "cross_reranking_*" | sort -t'_' -k3n | xargs -I {} sh -c 'grep --color=always -wH "P@1:" {} | awk "NR==3" '
```

Note: $NDCG@10$ shows most improvement.

### 3.1 Influence of Cross_Reranking with Query Summarization

First we need to summarize:
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --min_cluster_size=15 \
    --test_set_to_query \
    --logfile=query_creation
```

## 4. Influence of LLM Prompt-Reranking

```sh
# checking just the topic modelling predictions
launches/topic_modelling_zeroshot.sh \
--embedding_model_name=hkunlp/instructor-large \
--base_dir=../tmp_ablation/min_cluster_15 \
--source_url=XURLs.LF_AMAZON_131K_sample \
--dataset=deb101/lf_amazon_131k_sample \
--min_cluster_size=15 \
--predict \
--logfile=test_prompt_rerank

# checking the topic modelling + margin mse-loss predictions
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --train_margin_mseloss \
    --negatives_per_query=50 \
    --gpl_steps=7000 \
    --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
    --retrievers="['msmarco-distilbert-base-v3', 'hkunlp/instructor-large']" \
    --do_evaluation \
    --logfile=test_prompt_rerank2
```

Check logs so far

```sh
find . -name "test_prompt_rerank*" -type f -exec ls -lt {} + | awk '{print $9}' | xargs -I {} sh -c 'grep --color=always -wH "P@1:" {} && echo "*******"'
```

Now, to LLM rerank with ChatGPT add the --prompt_rerank:
```sh
export OPENAI_API_KEY="Your API key here"
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --min_cluster_size=15 \
    --predict \
    --prompt_rerank \
    --prompt_rerank_topk_lbs_cutoff=100 \
    --prompt_rerank_predicted_labels_col=agg_predicted_labels_sorted \
    --prompt_rerank_predicted_scores_col=agg_predicted_scores_sorted \
    --logfile=test_prompt_rerank_api_creation
```
This create files:
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation/min_cluster_15/LF_AMAZON_131K_sample$ find -name "api_requests*.jsonl"
```

Then launch the `api_rquests_part*.jsonl` files for LLM (ChatGPT) reranking using the batch script as follows:
```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$ 
launches/launch_openai_batches.sh ../tmp_ablation/min_cluster_15/LF_AMAZON_131K_sample/api_requests_part1.jsonl
``` 

This will create the following files:
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation/min_cluster_15/LF_AMAZON_131K_sample$ 
find -name "api_requests_part*_results.jsonl" 
./api_requests_part2_results.jsonl
./api_requests_part3_results.jsonl
./api_requests_part1_results.jsonl
```

Now to perform the reranking and computing the metrics based on the chatGPT results:
```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$
launches/rerank_gpt_preds.sh \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --labels_file_name="lbl.json" \
    --predicted_file="tst_predicted.csv" \
    --base_dir=../tmp_ablation/min_cluster_15
```

# Efficacy of our Approach in Few-Shot Setting

```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation/min_cluster_15/LF_AMAZON_131K_sample$ rm tst_predicted.csv
```
1. First train the topic modelling based model and get the predictions.

```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$
# Topic modelling predictions
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \    
    --base_dir=../tmp_ablation/min_cluster_15 \     
    --source_url=XURLs.LF_AMAZON_131K_sample \  
    --dataset=deb101/lf_amazon_131k_sample \ 
    --min_cluster_size=15 \     
    --predict \    
    --logfile=fewshot_topic_modelling
```

2. Next, Prepare the datasets in BeIR format for margin-mseloss training with:

```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --num_pos=10 \
    --prep_margin_mseloss_data \
    --label_frac_sup=3 \
    --logfile=fewshot_margin_mseloss_dataprep
```

3. Finally train (and evaluate) the Margin-MSELoss Model. 

```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name=hkunlp/instructor-large \
    --base_dir=../tmp_ablation/min_cluster_15 \
    --source_url=XURLs.LF_AMAZON_131K_sample \
    --dataset=deb101/lf_amazon_131k_sample \
    --train_margin_mseloss \
    --negatives_per_query=50 \
    --gpl_steps=1780 \
    --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
    --retrievers="['msmarco-distilbert-base-v3', 'hkunlp/instructor-large']" \
    --do_evaluation \
    --logfile=fewshot_margin_mseloss_train_eval
```

To perform the above steps 2 and 3 with a script to do ablation analysis (depicting the influence of supervision percentage `label_frac_sup`):

```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$ 
launches/fewshots.sh --label_frac_sup 1 2 3 6 --gpl_steps 7000 7000 7000 7000  
launches/fewshots.sh --label_frac_sup 10 --gpl_steps 10000
launches/fewshots.sh --label_frac_sup 18 32 70 --gpl_steps 15000 16000 25000
launches/fewshots.sh --label_frac_sup 56 100 --gpl_steps 20000 30000 #(needs to be run again)
```

To check the logs:
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp_ablation/min_cluster_15/LF_AMAZON_131K_sample$ 
# for 0
find logs -name "margin_mseloss_traineval_7614*" | sort -t'_' -k4n | xargs -I {} sh -c 'grep --color=always -wH "P@1:" {} && echo "*******"'

# for 1, 2, 3, 6
find logs -name "fewshot_margin_mseloss_train_eval_with_label_sup_*" | sort -t'_' -k9n | xargs -I {} sh -c 'grep --color=always -wH "P@1:" {} && echo "****
***"'

# for 10, 18, 32, 56, 70, 100
find -name "fewshot_margin_mseloss_train_eval_*_gpl*" | sort -t'_' -k6n | xargs -I {} sh -c 'grep --color=always -wH "P@1:" {} && echo "*******"'
```
