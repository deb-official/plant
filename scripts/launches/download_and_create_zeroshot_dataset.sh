#!/bin/bash

# Store all arguments in an array
args=(
--source_url="XURLs.LF_AMAZON_131K" 
--all_pairs_file_path="all_pairs.txt" 
--labels_file_path="lbl.json"
--train_file_path="trn.json"
--test_file_path="tst.json"
)

# Print all arguments
echo "All arguments:"
for arg in "${args[@]}"; do
    echo "$arg"
done   

python ./process_xmlzeroshot.py "${args[@]}"