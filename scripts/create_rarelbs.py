# import argparse
# import csv
# import pandas as pd
# from collections import Counter
# from pathlib import Path
from xcube.imports import *

def compute_label_frequencies(input_file, output_file, k=50, sort_by=None, label_delim=None):
    # Read CSV file into DataFrame
    df = pd.read_csv(input_file)
    if sort_by:
        df.sort_values(by=sort_by, inplace=True)#.reset_index(drop=True)

    # Compute label frequencies in the train and dev set
    lbl_freqs = Counter()
    for _,row in df.iterrows():
        if not row.is_valid and pd.notna(row.labels):
            lbl_freqs.update(row.labels.split(label_delim if label_delim is not None else ',' ))

    # Sort label frequencies in descending order
    lbls_sorted = sorted(lbl_freqs.items(), key=lambda item: item[1], reverse=True)

    # Save rarest `k` labels to CSV file
    rarest_labels = lbls_sorted[-k:]
    rarest_labels_list = [lbl for lbl,_ in rarest_labels]
    # import pdb; pdb.set_trace()
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "frequency"])
        for label, frequency in rarest_labels:
            writer.writerow([label, frequency])
    print(f"Rarest {k} labels saved to CSV file '{output_file}'.")
    
    # Select all datapoints that heave at least one of these rarest labels to create a rare dataset
    df_rare = df[ 
        df.labels.str.split(label_delim if label_delim is not None else ',' ).apply(
            lambda o: isinstance(o, list) and (True if set(o).intersection(lbl for lbl, _ in rarest_labels) else False)
        ) 
    ].reset_index(drop=True)
    # Filtering only rarest labels in df_rare
    df_rare.loc[:, 'labels'] = df_rare.labels.str.split(label_delim if label_delim is not None else ',' ).apply(
        lambda labels: label_delim.join([label for label in labels if label in rarest_labels_list])
    )

    input_path = Path(input_file)
    rare_fname = input_path.name.split('.')[0] + f'_rare{k}.' + input_path.name.split('.')[1]
    rare_fname = input_path.parent/rare_fname
    df_rare.to_csv(rare_fname)
    print(f"Rare datset of size {len(df_rare)} based off of {k} rareset labels saved to CSV file '{rare_fname}'.")
    # import pdb; pdb.set_trace()
    trn_sz, tst_sz = df_rare['is_valid'].value_counts()
    print(f"Train-Test split in the rare dataset is {trn_sz}-{tst_sz}.")

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Compute label frequencies on the training set and save rarest 50 labels to a CSV file.")
    # Add arguments
    parser.add_argument("--input_file", help="Path to the input CSV file.")
    parser.add_argument("--k", help="rarest k labels."),
    parser.add_argument("--sort_by", help="Sort by field in the input CSV file."),
    parser.add_argument("--output_file", help="Path to the output CSV file containing the rarest `k` labels and their frequencies.")
    parser.add_argument("--label_delim", help="label delimiter in the input file.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Compute label frequencies and save rarest labels to CSV file and create a rare dataset based off of rareest labels
    compute_label_frequencies(args.input_file, args.output_file, int(args.k), args.sort_by, args.label_delim)


if __name__ == "__main__":
    main()