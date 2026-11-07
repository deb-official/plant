import argparse
import csv
from tqdm import tqdm

def create_csv(train_text_file, train_labels_file, test_text_file, test_labels_file, output_file):
    # Read training texts from train_text_file
    with open(train_text_file, "r") as file:
        train_texts = file.readlines()

    # Read training labels from train_labels_file
    with open(train_labels_file, "r") as file:
        train_labels = file.readlines()

    # Remove newline characters and split labels by space
    train_labels = [line.strip().split() for line in train_labels]

    # Read test texts from test_text_file
    with open(test_text_file, "r") as file:
        test_texts = file.readlines()

    # Read test labels from test_labels_file
    with open(test_labels_file, "r") as file:
        test_labels = file.readlines()

    # Remove newline characters and split labels by space
    test_labels = [line.strip().split() for line in test_labels]

    # Write data to CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "labels", "is_valid"])  # Write header for CSV file
        # Write training data
        for text, labels in tqdm(zip(train_texts, train_labels), total=len(train_texts)):
            writer.writerow([text.strip(), ",".join(labels), "False"])
        # Write test data
        for text, labels in tqdm(zip(test_texts, test_labels), total=len(test_texts)):
            writer.writerow([text.strip(), ",".join(labels), "True"])

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Create CSV file from train and test text and labels.")
    # Add arguments
    parser.add_argument("--train_text_file", help="Path to the file containing training texts.")
    parser.add_argument("--train_labels_file", help="Path to the file containing training labels.")
    parser.add_argument("--test_text_file", help="Path to the file containing test texts.")
    parser.add_argument("--test_labels_file", help="Path to the file containing test labels.")
    parser.add_argument("--output_file", help="Path to the output CSV file.")

    # Parse the command-line arguments
    args = parser.parse_args()
    # import pdb; pdb.set_trace()

    # Create CSV file
    create_csv(args.train_text_file, args.train_labels_file, args.test_text_file, args.test_labels_file, args.output_file)

    print(f"CSV file '{args.output_file}' created successfully.")

if __name__ == "__main__":
    main()