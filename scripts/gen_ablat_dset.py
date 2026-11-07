#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: my_script.py
Description: This script demonstrates how to use argparse to take inputs from the command line. It performs basic operations based on the provided arguments.
Author: Your Name
Created Date: YYYY-MM-DD
Last Modified Date: YYYY-MM-DD
Version: 1.0
Usage: python topics.py --sum 10 20
"""

# Import standard libraries
import os
import sys
import argparse
import logging
from typing import List, Union

# Import xcube libraries
from xcube.data.all import *
from xcube.imports import *

# Constants
CONSTANT_EXAMPLE = 'Hello World'

# Function definitions
def generate_ablat_dset(
    source_url: str,
    fraction: float,
    label_coverage: list = [0, 1, 2, 3, 6, 10, 18, 32, 56, 70, 100]
    ):
    """
        Updated generate_ablat_dset with label coverage functionality.

        Parameters:
        - source_url (str): URL or path to the dataset directory (untarred dataset location).
        - fraction (float): Fraction of the dataset to sample (e.g., 0.1 for 10%).
        - label_coverage (list): List of label coverage values for sampling (e.g., [0, 1, 2, 3, 5, ...]).

        Returns:
        - Path: Path to the directory containing the sampled dataset and JSON files.
        """ 
    
    source = untar_xxx(eval(source_url))

    # Log the start of the operation
    logging.info(f"Starting to sample from dataset at {source} with a fraction of {fraction}")
    
    # Find the first CSV file in the source directory
    dset_file = list(source.glob("*.csv"))[0]
    logging.info(f"Found dataset file: {dset_file.name}")
    
    # Read the dataframe from the CSV file
    df = pd.read_csv(dset_file, usecols=['uid', 'title', 'content', 'target_ind', 'labels', 'split'])
    logging.info(f"Read dataframe with {len(df)} rows.")
    
    # Separate the dataframe into 'train' and 'test' splits
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    logging.info(f"Separated dataset into {len(train_df)} train rows and {len(test_df)} test rows.")
    
    # Sample the 'train' and 'test' data independently while maintaining the same ratio
    train_sample = train_df.sample(frac=fraction, random_state=42)
    test_sample = test_df.sample(frac=fraction, random_state=42)
    logging.info(f"Sampled {len(train_sample)} rows from train and {len(test_sample)} rows from test data.")
    
    # Concatenate the sampled 'train' and 'test' data
    sampled_df = pd.concat([train_sample, test_sample])

    # Create a new directory for the sampled data by adding '_sample' to the source path
    source_sample = (source.parent) / (source.name + '_sample')
    
    # Create the new directory if it doesn't exist
    os.makedirs(source_sample, exist_ok=True)
    logging.info(f"Created directory: {source_sample}")
    
    # Save the sampled dataframe to the new directory, using the original CSV file name
    sampled_file = source_sample / dset_file.name
    sampled_df.to_csv(sampled_file, index=False)
    logging.info(f"Saved sampled dataframe to: {sampled_file}")

    # Create separate lists of uids for train and test
    trn_sampled_uids = train_sample['uid'].tolist()
    tst_sampled_uids = test_sample['uid'].tolist()

    # Read the trn.json and tst.json files
    trn_file = source / 'trn.json'
    tst_file = source / 'tst.json'
    
    # Load the JSON data
    # Filter the records from trn.json and tst.json based on the sampled uids
    with open(trn_file, 'r') as f:
        trn_sampled_data = []
        for line in f:
            record = json.loads(line)
            if record['uid'] in trn_sampled_uids: trn_sampled_data.append(record)
    with open(tst_file, 'r') as f:
        tst_sampled_data = []
        for line in f:
            record = json.loads(line)
            if record['uid'] in tst_sampled_uids: tst_sampled_data.append(record)
    
    logging.info(f"Loaded {len(trn_sampled_data)} records from trn.json and {len(tst_sampled_data)} records from tst.json.")

    # Save the filtered JSON data to the sample directory
    trn_sample_file = source_sample / 'trn.json'
    tst_sample_file = source_sample / 'tst.json'

    with open(trn_sample_file, 'w') as f:
        for record in trn_sampled_data:
            json.dump(record, f)
            f.write('\n')  # Write each record on a new line
    with open(tst_sample_file, 'w') as f:
        for record in tst_sampled_data:
            json.dump(record, f)
            f.write('\n')  # Write each record on a new line
    
    logging.info(f"Saved sampled JSON data to: {trn_sample_file} and {tst_sample_file}")

    # Create a set of all unique label uids from the 'labels' column in sampled_df
    sampled_df['labels'] = sampled_df['labels'].apply(ast.literal_eval)
    label_uids = set()
    for labels_dict in sampled_df['labels']:
        label_uids.update(labels_dict.keys())
    
    logging.info(f"Found {len(label_uids)} unique labels in sampled_df.")

    # Read the lbl.json file
    lbl_file = source / 'lbl.json'
    
    # Load the JSON data from lbl.json
    with open(lbl_file, 'r') as f:
        lbl_data = []
        for line in f:
            record = json.loads(line)
            if record['uid'] in label_uids: lbl_data.append(record)

    logging.info(f"Loaded {len(lbl_data)} records from lbl.json.")
    
    # Save the filtered JSON records to a new JSONL file
    filtered_lbl_file = source_sample / 'lbl.json'
    
    with open(filtered_lbl_file, 'w') as f:
        for record in lbl_data:
            json.dump(record, f)
            f.write('\n')  # Write each record on a new line
    
    logging.info(f"Saved filtered label JSONL data to: {filtered_lbl_file}")

    # Process label coverage
    for l in label_coverage:
        if l > len(label_uids):
            logging.warning(f"Label coverage {l} exceeds total unique labels ({len(label_uids)}). Skipping.")
            continue
        
        # Random-sample l fraction of labels
        label_uids_coverage_l = set(pd.Series(list(label_uids)).sample(frac=float(l/100), random_state=42))
        logging.info(f"Selected {len(label_uids_coverage_l)} out of total labels {len(label_uids)}. Coverage Fraction: {l}%.")
        
        # Gather sampled_df['uid'] where any label in 'labels' exists in label_uids_coverage_l
        train_uids_l = sampled_df[
            (sampled_df['split'] == 'train') & 
            (sampled_df['labels'].apply(lambda x: bool(set(x.keys()) & label_uids_coverage_l)))
        ]['uid'].tolist()
        
        # Save the list of UIDs to a file
        train_uids_l_file = source_sample / f"train_uids_{l}.txt"
        with open(train_uids_l_file, 'w') as f:
            f.writelines(f"{uid}\n" for uid in train_uids_l)
        logging.info(f"Saved {len(train_uids_l)} UIDs for label coverage {l} to {train_uids_l_file}")

    return source_sample

def main():
    """
    Main function to control the program execution. Parses arguments and calls the appropriate process functions.
    """

    # Setup argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--source_url", type=str, help="URL of the dataset")
    parser.add_argument('--fraction', type=float, default=0.2, help='Sample fraction default is 0.2.')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Program started")

    try:
        # Process data based on arguments
        generate_ablat_dset(**vars(args))

    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        sys.exit(1)
    
    logging.info("Program ended successfully")

# Ensure the script can be used as a script as well as an importable module
if __name__ == "__main__":
    main()
