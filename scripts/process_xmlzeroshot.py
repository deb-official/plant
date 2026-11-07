from fastai.distributed import *
from xcube.imports import *
from xcube.data.all import *

def process_pairs(file_path):
    qrels = {}

    with open(file_path, 'r') as f:
        for line in f:
            import pdb; pdb.set_trace()
            query_id, corpus_id, score = line.strip().split()
            score = int(score)
            if query_id not in qrels:
                qrels[query_id] = {corpus_id: score}
            else:
                qrels[query_id][corpus_id] = score

    return qrels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pairs file")
    parser.add_argument("--source_url", type=str, help="URL of the dataset")
    parser.add_argument("--all_pairs_file_path", type=str, help="Path to the file containing pairs")
    parser.add_argument("--labels_file_path", type=str, help="Path to the file containing label descriptions")
    parser.add_argument("--train_file_path", type=str, help="Path to the file containing training examples")
    parser.add_argument("--test_file_path", type=str, help="Path to the file containing test examples")

    args = parser.parse_args()

    source = rank0_first(untar_xxx, eval(args.source_url))

    # Check if any argument is None
    if any(arg is None for arg in vars(args).values()):
        # Print a message informing the user to provide all arguments
        print("Please provide values for all arguments (--source_url ,--all_pairs_file_path, --labels_file_path, --train_file_path, --test_file_path)")
        # Exit the program
        sys.exit(1)
    
    # Create an empty dictionary to store label descriptions
    descriptions, labels_records = {}, []
    # Load the lbl.json file
    with open(source/args.labels_file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            labels_records.append(record)
            uid = record['uid']
            descriptions[uid] = record['title']  # Change this to whatever description field you want to extract

    # Create a mapping from index to label records for efficient lookup
    label_map = {i: {"uid": rec["uid"], "title": rec["title"]} for i, rec in enumerate(labels_records)}

    def map_labels(target_indices):
        # Function to map target_ind list to label dicts
        return {label_map[idx]["uid"]: label_map[idx]["title"] for idx in target_indices if idx in label_map}

    # process all_pairs to compute the related labels for each uid
    # qrels = process_pairs(source/args.all_pairs_file_path)
    # for uid in qrels:
    #     for related_uid in qrels[uid]:
    #         qrels[uid][related_uid] = descriptions.get(related_uid, None)

    # Initialize an empty list to store the JSON records
    data = []
    # Read each line of the JSON train file and parse it as a JSON object
    with open(source/args.train_file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            record['split'] = 'train'
            data.append(record)
    # Read each line of the JSON test file and parse it as a JSON object
    with open(source/args.test_file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            record['split'] = 'test'
            data.append(record)       
    # Convert the list of JSON records to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Iterate over DataFrame rows to add labels
    # uids = df['uid']
    # labels = uids.map(qrels.get)
    # df['labels'] = labels
    # Add the 'labels' column
    df["labels"] = df["target_ind"].apply(map_labels)

    # Save the DataFrame to a CSV file
    dataset_name = str.lower(source.name)
    file = source/str.lower(dataset_name+'.csv')
    print(f"Saving data in file {file}...")
    df.to_csv(file, index=False)