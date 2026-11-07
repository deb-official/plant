# LF-WikiSeeAlso-320K

##  Prepare the dataset

```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$
launches/download_and_create_zeroshot_dataset_new.sh \
    --source_url=XURLs.LF_WIKISEEALSO_320K
```

This will create the files:
```sh
(base) deb@deb-GIGABYTE:~/.xcube/data$ l LF-WikiSeeAlso-320K/
all_pairs.txt  lbl.json  lf-wikiseealso-320k.csv  trn.json  tst.json
```

```sh
(base) deb@deb-GIGABYTE:~/.xcube/data$
tar -czf LF-WikiSeeAlso-320K.tar.gz LF-WikiSeeAlso-320K/
aws s3 cp LF-WikiSeeAlso-320K.tar.gz s3://xcubebucket/lf-wikiseealso-320k/
```

Also, upload the above generated files to huggingface dataset. Make sure to update the README.md reflecting the split information. 

## Topic Aware Semantic Search (TASS)

Train the topic model and topicfy the data:
Notes: 
1. comment the openai_api key in topic_modelling_zeroshot.sh)
2. '/home/deb/miniconda3/lib/python3.10/site-packages/bertopic/_bertopic.py' line 3989 in `_extract_topics` - truncating docs before aggregation (hack to make the scikit learn's `CountVectorizer` work on huge XMTC datasets)

```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K \
    --dataset=deb101/lf_wikiseealso_320k \
    --min_cluster_size=120 \
    --min_topic_size=10 \
    --logfile=topic_model_train \
    --no_fresh_topicmodel
```

This will save the topic model at `~/xcube/tmp2/LF_WIKISEEALSO_320K` and generate files:
```sh
deb@deb-GIGABYTE:~/xcube/tmp2/LF_WIKISEEALSO_320K$ find -name "*.csv"
./topics.csv
./topic_info.csv
./trn_clustered.csv ## This is the topicfied training data
```

Now we need to pseudo-label the data, by adding the flag `--pseudolabel`:
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K \
    --dataset=deb101/lf_wikiseealso_320k \
    --min_cluster_size=120 \
    --min_topic_size=10 \
    --no_fresh_topicmodel \
    --pseudolabel \
    --logfile=topic_model_pseudolabel
```

This will take as input the previously generated `trn_clustered.csv` file and produce:
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp2/LF_WIKISEEALSO_320K$ find -name "*.csv" -mmin -15
./topicfied_labels.csv # The topicfied labels

./trn_labels_for_each_cluster_by_topicfying_labels.csv # The label set for each topic cluster of the labels (label regularization)

./trn_labels_for_each_cluster.csv # The label set for each topic cluster

./trn_clustered_pseudo_labelled.csv # This is the train_clustered.csv file but now pseudo labelled
```

Now, the final step in this **Topic Modelling** stage is to make the predictions using the `--predict` flag:
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K \
    --dataset=deb101/lf_wikiseealso_320k \
    --min_cluster_size=120 \
    --min_topic_size=10 \
    --no_fresh_topicmodel \
    --train_test_file_path="lf-wikiseealso-320k.csv" \ # make sure you give the correct file
    --predict \
    # --no_topic_labels \ # use this you want full corpus semantic search
    --logfile=topic_model_predict_120
```

you can use the following script to hyperparameter search the best `min_cluster_size`
```sh
launches/topics_ablation_wikiseealso-320k.sh --min_cluster_size 120
```
The current best log file (with full corpus vanilla semantic search):
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp2/LF_WIKISEEALSO_320K$ 
ls logs/topic_model_predict_120_2024-12-04_13-08-05.log
```

## Margin-MSELoss Training

1. Prepare the datasets in BeIR format for margin-mseloss training with:

```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K \
    --dataset=deb101/lf_wikiseealso_320k \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    # --do_summarize \ # use this if you want to summarize the queries but this takes a long time
    --num_pos=20 \
    --max_content_length=250 \ # This is the truncation length for the beir format queries of the train and test set
    --prep_margin_mseloss_data \
    --logfile=margin_mseloss_dataprep
```

This will generate the following files:
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp2/LF_WIKISEEALSO_320K$ find -type f -mmin -30
./corpus.jsonl # These are the labels
./qgen-queries.jsonl # These are the queries of the training set
./qgen-qrels/train.tsv # These are the relation. For each query_id there are `num_pos` many positives
./qrels/test.tsv # These are ground truth relations for the test set queries
./queries.jsonl # These are the queries of the test set
```

2. Now to perform the margin-mseloss training data generation with:

You can compute the `gpl_steps` that you need for a desired number of queries per entry of the training data by the script. Here `num_train_queries` is the size of the training set. 
```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$ ./calculate_gpl_steps.sh
Usage: ./calculate_gpl_steps.sh <num_train_queries> <desired_average_entries>
num_train_queries=$(wc -l ~/.xcube/data/LF-WikiSeeAlso-320K/trn.json | awk -F' ' '{print $1}')
bash calculate_gpl_steps.sh $num_train_queries 10
```
Note: You can also set a different `cross_encoder`. This will be used in pseudo-labelling the negative examples to create `gpl-training-data.tsv`

```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K \
    --dataset=deb101/lf_wikiseealso_320k \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    --train_margin_mseloss \
    --negatives_per_query=50 \
    --gpl_steps=216600 \
    --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
    --retrievers="['sentence-transformers/msmarco-distilbert-base-tas-b', 'hkunlp/instructor-large']" \
    --cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --logfile=margin_mseloss_train_data_prep \
    --stop_after_gpl_train_data_creation
```
This will generate `hard-negatives.jsonl` and `gpl-training-data.tsv`.

```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp2/LF_WIKISEEALSO_320K$ ls -t | head -n 2
gpl-training-data.tsv
hard-negatives.jsonl
```
From the Train set select a query and look at the positives and negative labels generated. 
- The positive labels are generated by **TASS**,
- The negative labels are generated by negative miners specified by the flag `retrievers`
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp2/LF_WIKISEEALSO_320K$ 
# To look at the ground truth
grep '^Clochette,' /home/deb/.xcube/data/LF-WikiSeeAlso-320K/lf-wikiseealso-320k.csv
# awk -F, '$1 == "Clochette"'  

# To look at the training query correspond to the ground truth
cat qgen-queries.jsonl | jq 'select(._id=="Clochette")'

# To look at the positive and negative labels
cat hard-negatives.jsonl | jq 'select(.qid=="Clochette")'
```

Once we have the `gpl-training-data.tsv`, we can perform training and evaluation just remove the flag `stop_after_gpl_train_data_creation` add the flag `do_evaluation`:
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K \
    --dataset=deb101/lf_wikiseealso_320k \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    --train_margin_mseloss \
    --negatives_per_query=50 \
    --gpl_steps=216600 \
    --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
    --retrievers="['sentence-transformers/msmarco-distilbert-base-tas-b', 'hkunlp/instructor-large']" \
    --cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --do_evaluation \
    --logfile=margin_mseloss_traineval
```

Notes:
/home/deb/gpl/gpl/toolkit/evaluation.py `chunk_size` = 30000
/home/deb/miniconda3/lib/python3.10/site-packages/beir/retrieval/search/dense/exact_search.py

with 
--base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
P@1: 15.31 (just margin loss) and 21.54 (aggregated with TASS)

Training one more epoch:
change base_ckpt from
--base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
to
--base_ckpt="./output/216600_old" to load the model that was trained for one epoch


# LF-WikiSeeAlso-320K_sample

## Preparing the sample

The following will generate a sample of the dataset
```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$ ./launches/generate_ablat.sh
```

```sh
(base) deb@deb-GIGABYTE:~/.xcube/data/LF-WikiSeeAlso-320K_sample$ tree
.
├── lbl.json
├── lf-wikiseealso-320k.csv
├── trn.json
└── tst.json

0 directories, 4 files
cd ..
tar -czf LF-WikiSeeAlso-320K_sample.tar.gz LF-WikiSeeAlso-320K_sample/
aws s3 mv LF-WikiSeeAlso-320K_sample.tar.gz s3://xcubebucket/lf-wikiseealso-320k/
```

Also, upload the above generated files to huggingface dataset. Make sure to update the README.md reflecting the split information.

```
(base) deb@deb-GIGABYTE:~/.xcube/data/LF-WikiSeeAlso-320K_sample$ 
huggingface-cli login # login with write token
huggingface-cli repo create lf_wikiseealso_320k_sample --type dataset
git lfs install   # Only needed once if you haven't set up Git LFS already
git clone https://huggingface.co/datasets/your-username/LF-WikiSeeAlso-320K_sample
cp -r /path/to/LF-WikiSeeAlso-320K_sample/* LF-WikiSeeAlso-320K_sample/
# Update README.md to add a dataset split information and dataset card
git add .
```
Your dataset is now live on the Hugging Face Hub at:
https://huggingface.co/datasets/your-username/LF-WikiSeeAlso-320K_sample

## Topic Aware Semantic Search (TASS)

**Step 1:** Train the topic model and topicfy the data:
```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K_sample \
    --dataset=deb101/lf_wikiseealso_320k_sample \
    --min_cluster_size=120 \
    --min_topic_size=10 \
    --logfile=topic_model_train
```

**Step 2:** Now we need to pseudo-label the data, by adding the flag `--pseudolabel`:
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K_sample \
    --dataset=deb101/lf_wikiseealso_320k_sample \
    --min_cluster_size=120 \
    --min_topic_size=10 \
    --pseudolabel \
    --logfile=topic_model_pseudolabel
```

**Step 3:** Now, the final step in this **Topic Modelling** stage is to make the predictions using the `--predict` flag:
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K_sample \
    --dataset=deb101/lf_wikiseealso_320k_sample \
    --min_cluster_size=120 \
    --min_topic_size=10 \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    --predict \
    --logfile=topic_model_predict_120
```

## Margin-Augmented Dense Retriever for Semantic Similarity (MARS)

**Step 1.** Prepare the datasets in BeIR format for margin-mseloss training with:

(optionally export the `OPENAI_API_KEY` if you also want to generate summarization requests for CHATGPT)
```sh
export OPENAI_API_KEY="Your API key here"
```
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K_sample \
    --dataset=deb101/lf_wikiseealso_320k_sample \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    # --do_summarize \ # use this if you want to summarize the queries but this takes a long time
    --num_pos=5 \
    --max_content_length=250 \ # This is the truncation length for the beir format queries of the train and test set
    --prep_margin_mseloss_data \
    --logfile=margin_mseloss_dataprep
```
If you had the `OPENAI_API_KEY` set, the above in addition to preparing the BeIR format datasets, will also create files `find -type f -name "train*summarization*.jsonl"` and `find -type f name "test*summarization*.jsonl"` in the `base_dir`. You can launch the summarization requests to CHATGPT using the following script:
```sh
launches/launch_openai_batches.sh ../tmp2/LF_WIKISEEALSO_320K_sample/train_summarization_requests_part1.jsonl
```
The above script will save the output returned by CHATGPT in files `train_summarization_requests_part1_results.jsonl`. To check what CHATGPT returns:
```sh
launches/random_summarization_menu.sh ../tmp2/LF_WIKISEEALSO_320K_sample/train_summarization_requests_part1_results.jsonl
```

**Step 1a:** After you have launched all the train and test summarization and also gotten back the results, you can collate the CHATGPT summarized queries using the following command by using the `chatgpt_query_summary` flag:
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K_sample \
    --dataset=deb101/lf_wikiseealso_320k_sample \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    --num_pos=5 \
    --max_content_length=250 \ # This is the truncation length for the beir format queries of the train and test set
	--chatgpt_query_summary \
    --logfile=margin_mseloss_chatgpt_query_summary
```
This will  save the train queries to `../tmp2/LF_WIKISEEALSO_320K_sample/train_chatgpt_queries.jsonl` and the test queries to `../tmp2/LF_WIKISEEALSO_320K_sample/test_chatgpt_queries.jsonl`. Before proceeding to Step2 you should rename these files to `qgen-queries.jsonl` and `queries.jsonl` if you want to use the CHATGPT queries in gpl training. 

**Step 2:** Now to perform the margin-mseloss training data generation with:
```sh
num_train_queries=$(wc -l ~/.xcube/data/LF-WikiSeeAlso-320K_sample/trn.json | awk -F' ' '{print $1}')
bash calculate_gpl_steps.sh $num_train_queries 10
To achieve an average of 10 entries per query with 138616 queries, you should use approximately 43317 GPL steps.
```
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K_sample \
    --dataset=deb101/lf_wikiseealso_320k_sample \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    --train_margin_mseloss \
    --negatives_per_query=50 \
    --gpl_steps=43317 \
    --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
    --retrievers="['sentence-transformers/msmarco-distilbert-base-tas-b', 'hkunlp/instructor-large']" \
    --cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --logfile=margin_mseloss_train_data_prep \
    --stop_after_gpl_train_data_creation
```
This will generate `hard-negatives.jsonl` and `gpl-training-data.tsv`.

**Step 3:** Once we have the `gpl-training-data.tsv`, we can perform training and evaluation just remove the flag `stop_after_gpl_train_data_creation` add the flag `do_evaluation`:

```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K_sample \
    --dataset=deb101/lf_wikiseealso_320k_sample \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    --train_margin_mseloss \
    --negatives_per_query=50 \
    --gpl_steps=43317 \
    --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
    --retrievers="['sentence-transformers/msmarco-distilbert-base-tas-b', 'hkunlp/instructor-large']" \
    --cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --do_evaluation \
    --logfile=margin_mseloss_traineval \
```

Note: You can modify the `checkpoint_save_steps` to keep record of intermediate gpl_steps, we changed it to 1000, default is 10000. After it is done you can run the evaluation at each of those intermediate steps by:

```sh
# run the evaluations (change the looping in the script)
bash launches/launch_margin_mseloss_intermediate_gpl_steps_eval.sh

# view the logs (margin-mse performance on the 1st line and TASS + margin-mse aggregation on the 2nd line)
find logs -name "margin_mseloss_traineval_numpos5_*_2025*" | sort -t'_' -k4n | xargs -I {} sh -c 'grep --color=always -wH "P@1:" {} && echo "*******"'

# view just the margin-msel aggregation (1st line)
find logs -name "margin_mseloss_traineval_numpos5_*_2025*" | sort -t'_' -k5n | xargs -I {} sh -c 'grep --color=always -wH "P@1:" {} |  awk "NR==2"'
```

Note: During the data generation:

| `num_pos` | $P@1$                       |
| --------- | --------------------------- |
| 20        | 34.90                       |
| 10        | 35.31                       |
| 5         | 36.08 (just margin-mseloss) |
## Cross-Reranking

First, make sure we have done the stages (1) TASS and (2) MARS.

Note: Make sure to remove
```sh 
(base) deb@deb-GIGABYTE:~/xcube/tmp2/LF_WIKISEEALSO_320K_sample$ rm tst_predicted.csv
```
```sh
# check the TASS performance
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K_sample \
    --dataset=deb101/lf_wikiseealso_320k_sample \
    --min_cluster_size=120 \
    --min_topic_size=10 \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    --predict \
    --logfile=topic_model_predict_120

#check the MARS performance
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K_sample \
    --dataset=deb101/lf_wikiseealso_320k_sample \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    --train_margin_mseloss \
    --negatives_per_query=50 \
    --gpl_steps=43317 \
    --base_ckpt="GPL/cqadupstack-tsdae-msmarco-distilbert-gpl" \
    --retrievers="['sentence-transformers/msmarco-distilbert-base-tas-b', 'hkunlp/instructor-large']" \
    --cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --do_evaluation \
    --logfile=margin_mseloss_traineval
```

Then we can do the **cross reranking**.
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K_sample \
    --dataset=deb101/lf_wikiseealso_320k_sample \
    --min_cluster_size=120 \
    --min_topic_size=10 \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    --predict \
    --predict_topk=13 \
    --cross_rerank \
    --logfile="cross_reranking_13"
```

Note: The file 'tst_predicted.csv' has a bunch of columns. The following columns have the meaning:

| index | column_names                              | column_meaning                              |
| ----- | ----------------------------------------- | ------------------------------------------- |
| 1     | `predicted_scores`                        | TASS predictions                            |
| 2     | `margin_mseloss_predicted_scores`         | MARS predictions                            |
| 3     | `agg_predicted_scores`                    | aggregation of TASS & MARS                  |
| 4     | `predicted_scores_reranked`               | aggregation of MARS and cross-reranked MARS |
| 5     | `topic_margin_cross_agg_predicted_scores` | aggregation of 3 and 4                      |

```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp2/LF_WIKISEEALSO_320K_sample$ head -n 1 tst_predicted.csv | tr ',' '\n'
```
We will use the columns `topic_margin_cross_agg_predicted_scores` and `topic_margin_cross_agg_predicted_labels` in the next section to LLM Prompt-Rerank.
## Efficient LLM Prompt-Reranking

Now, to LLM rerank with ChatGPT add the `--prompt_rerank` flag and use the columns `topic_margin_cross_agg_predicted_scores` and `topic_margin_cross_agg_predicted_labels` that were generated in the previous section (cross-reranking) 
```sh
export OPENAI_API_KEY="Your API key here"
```
```sh
launches/topic_modelling_zeroshot.sh \
    --embedding_model_name="sentence-transformers/msmarco-distilbert-base-tas-b" \
    --base_dir=../tmp2 \
    --source_url=XURLs.LF_WIKISEEALSO_320K_sample \
    --dataset=deb101/lf_wikiseealso_320k_sample \
    --min_cluster_size=120 \
    --min_topic_size=10 \
    --train_test_file_path="lf-wikiseealso-320k.csv" \
    --predict \
    --prompt_rerank \
    --prompt_rerank_topk_lbs_cutoff=50 \
    --prompt_rerank_predicted_labels_col=topic_margin_cross_agg_predicted_labels \
    --prompt_rerank_predicted_scores_col=topic_margin_cross_agg_predicted_scores \
    --max_records_per_file=10000 \
    --max_doc_length_chatgpt_ranking=300 \
    --logfile=test_prompt_rerank_api_creation
```
This create files:
```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp2/LF_WIKISEEALSO_320K_sample$ find -name "api_requests*.jsonl"

# You can check out the user message and the prompts
shuf -n 1 ./api_requests_part1.jsonl | jq -r '.messages[] | select(.role=="system" or .role=="user") | "\(.role) content: \(.content)\n"'
```

Then launch the `api_rquests_part*.jsonl` files for LLM (ChatGPT) reranking using the batch script as follows:

```sh
deb@deb-GIGABYTE:~/xcube/scripts$ launches/launch_openai_batches.sh ../tmp2/LF_WIKISEEALSO_320K_sample/api_requests_part1.jsonl
```
This will create the result files (that is the response of the API requests). These would be the files: 

```sh
(base) deb@deb-GIGABYTE:~/xcube/tmp2/LF_WIKISEEALSO_320K_sample$ find -name "api_requests*results.jsonl"
```
Now, you pick one of the returned file from above and check the prompt and CHATGPT's response 
```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$ 
./launches/extract_chatprompt.sh --filename /home/deb/xcube/tmp2/LF_WIKISEEALSO_320K_sample/api_requests_part1_results.jsonl --requestid 216
```

Finally, to perform the reranking and computing the metrics based on the chatGPT results:

```sh
launches/rerank_gpt_preds.sh \
	--source_url=XURLs.LF_WIKISEEALSO_320K_sample \
	--labels_file_name="lbl.json" \
	--label_mapping_file="label_uid_mappings.csv" \
	--predicted_file="tst_predicted.csv" \
	--base_dir=../tmp2 \
	--fresh
```

