#!/bin/bash

# Store all arguments in an array
args=(
--source_url="XURLs.LF_WIKISEEALSO_320K" 
# --all_pairs_file_path="all_pairs.txt" 
# --labels_file_path="lbl.json"
# --train_file_path="trn.json"
# --test_file_path="tst.json"
--fraction="0.2"
)

# Print all arguments
echo "All arguments:"
for arg in "${args[@]}"; do
    echo "$arg"
done   

python ./gen_ablat_dset.py "${args[@]}"