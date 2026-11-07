from xcube.imports import *

def compute_label_frequencies(input_file, output_file, sort_by=None, label_delim=None):
    # Read CSV file into DataFrame
    df = pd.read_csv(input_file)

    if sort_by:
        df.sort_values(by=sort_by, inplace=True)#.reset_index(drop=True)

    def code_freq(df):
        code_frqs = Counter()
        for labels in df.labels:
            if pd.notna(labels): code_frqs.update(labels.split(label_delim if label_delim is not None else ','))
        return code_frqs

    test_set_codes = code_freq(df[df.is_valid])
    print(f"Number of unique ICD codes in the test set: {len(test_set_codes)}")

    train_set_codes = code_freq(df[df['split'] == 'train'])
    print(f"Number of unique ICD codes in the train set: {len(train_set_codes)}")

    rare_codes = [(code,frq) for code,frq in test_set_codes.items() if code in train_set_codes and train_set_codes[code]>=1 and train_set_codes[code]<=5]
    rare_codes_list = [code for code,_ in rare_codes]
    print(f"Number of rare codes (of all the test set codes the ones that occur at least once and at most 5 times in the training set) = { len(rare_codes) }")

    # Save rare codes to CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "frequency"])
        for (code, frq) in rare_codes:
            writer.writerow([code, frq])
        print(f"The rare codes have been saved to the CSV file '{output_file}'.")

    # Select all datapoints that heave at least one of these rare codes to create a rare dataset
    df_rare = df[ 
        df.labels.str.split(label_delim if label_delim is not None else ',').apply(
            lambda o: isinstance(o, list) and (True if set(o).intersection(rare_codes_list) else False)
        ) 
    ].reset_index(drop=True)
    # Filtering only rarest labels in df_rare
    df_rare.loc[:, 'labels'] = df_rare.labels.str.split(label_delim if label_delim is not None else ',').apply(
        lambda labels: ';'.join([label for label in labels if label in rare_codes_list])
    )

    input_path = Path(input_file)
    rare_fname = input_path.name.split('.')[0].split('_')[0] + f'_few.' + input_path.name.split('.')[1]
    rare_fname = input_path.parent/rare_fname
    df_rare.to_csv(rare_fname)
    print(f"Rare datset of size {len(df_rare)} based off of rare codes saved to CSV file '{rare_fname}'.")
    value_counts = df_rare['split'].value_counts()
    print(f"Split sizes in the rare dataset is: \n{value_counts.to_string(header=False)}.")

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Among unique ICD codes in the test set, collect those that occur less than five times (but at least once) in the training set. We will call those rare labels. And, save those rare labels to a CSV file.")
    # Add arguments
    parser.add_argument("--input_file", help="Path to the input CSV file.")
    parser.add_argument("--sort_by", help="Sort by field in the input CSV file."),
    parser.add_argument("--output_file", help="Path to the output CSV file containing the rare labels and their frequencies.")
    parser.add_argument("--label_delim", help="label delimiter in the input file.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Compute label frequencies and save rarest labels to CSV file and create a rare dataset based off of rareest labels
    compute_label_frequencies(args.input_file, args.output_file, args.sort_by, args.label_delim)
    
if __name__ == "__main__":
    main()