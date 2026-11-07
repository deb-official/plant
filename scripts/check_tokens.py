#!/usr/bin/env python

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset
import logging
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, Any, Optional, List

from xcube.imports import *


def setup_logger(output_dir: str) -> logging.Logger:
    """
    Set up logging to a timestamped file and console.

    Creates a logfile named 'sample_tokenization_YYYYMMDD_HHMMSS.log' in the output directory
    and configures logging to write to both the file and console.

    Args:
        output_dir (str): Directory to save the logfile.

    Returns:
        logging.Logger: Configured logger instance.
    """
    global logger
    # if logger is not None:
    #     # Avoid re-initializing if already set
    #     logger.info(f"Logger already initialized, using existing logger.")
    #     pdb.set_trace()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for logfile in YYYY-MM-DD_HHMMSS format
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(output_dir, f"sample_tokenization_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger()
    logger.info(f"Logging initialized at {timestamp}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def get_data(source, data):
    """
    Load CSV data into a pandas DataFrame.

    Constructs a file path and reads the CSV file, ensuring 'text' and 'labels' columns are strings.

    Args:
        source (str): Directory containing the CSV file.
        data (str): Base filename (without extension) of the CSV data.

    Returns:
        pd.DataFrame: DataFrame with 'text', 'labels', and 'is_valid' columns.
    """
    # Construct file path (assuming CSV is in source directory)
    filepath = os.path.join(source, f"{data}.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    
    # Read CSV, selecting specified columns
    df = pd.read_csv(
        filepath,
        header=0,
        usecols=['text', 'labels', 'is_valid'],
        dtype={'text': str, 'labels': str, 'is_valid': bool}
    )
    
    # Ensure text and labels are strings
    df[['text', 'labels']] = df[['text', 'labels']].astype(str)
    
    return df


def chunk_text(text: str, tokenizer: AutoTokenizer, max_length: int = 512, overlap: int = 50):
    """
    Split text into chunks that fit within max_length tokens, with optional overlap.

    Args:
        text (str): Input text to chunk.
        tokenizer (AutoTokenizer): Tokenizer to encode text.
        max_length (int): Maximum tokens per chunk (including special tokens).
        overlap (int): Number of tokens to overlap between chunks.

    Returns:
        List[Dict]: List of tokenized chunks, each with input_ids, attention_mask, etc.
    """
    # Tokenize the entire text without truncation
    encoded = tokenizer(
        text,
        add_special_tokens=False,  # Add [CLS], [SEP] per chunk later
        return_tensors="pt",
        return_attention_mask=True
    )
    
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    
    # Account for [CLS] and [SEP] in max_length
    chunk_size = max_length - 2  # Reserve 1 for [CLS], 1 for [SEP]
    if chunk_size <= 0:
        raise ValueError("max_length must be at least 3 to accommodate special tokens")
    
    chunks = []
    i = 0
    while i < len(input_ids):
        # Determine chunk end, ensuring it doesn't exceed input length
        end = min(i + chunk_size, len(input_ids))
        
        # Extract chunk
        chunk_ids = input_ids[i:end]
        chunk_mask = attention_mask[i:end]
        
        # Add special tokens
        cls_id = tokenizer.cls_token_id or tokenizer.bos_token_id
        sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id
        chunk_ids = torch.cat([
            torch.tensor([cls_id], dtype=chunk_ids.dtype),
            chunk_ids,
            torch.tensor([sep_id], dtype=chunk_ids.dtype)
        ])
        chunk_mask = torch.cat([
            torch.tensor([1], dtype=chunk_mask.dtype),
            chunk_mask,
            torch.tensor([1], dtype=chunk_mask.dtype)
        ])
        
        # Pad to max_length if needed
        padding_length = max_length - len(chunk_ids)
        if padding_length > 0:
            chunk_ids = torch.cat([
                chunk_ids,
                torch.zeros(padding_length, dtype=chunk_ids.dtype)
            ])
            chunk_mask = torch.cat([
                chunk_mask,
                torch.zeros(padding_length, dtype=chunk_mask.dtype)
            ])
        
        # Store chunk
        chunks.append({
            "input_ids": chunk_ids,
            "attention_mask": chunk_mask,
            "token_type_ids": torch.zeros_like(chunk_ids) if "token_type_ids" in tokenizer.model_input_names else None
        })
        
        # Move to next chunk with overlap
        i += chunk_size - overlap
    
    return chunks

class FlatChunkDataset(Dataset):
    """
    A PyTorch Dataset that treats each chunk as an individual item, with processed labels and one-hot encodings.
    """
    def __init__(self, dataset: Dataset, label_delim: Optional[str] = None):
        self.items = []
        for idx, item in enumerate(dataset):
            chunks = item["chunks"]
            if "labels" in item:
                label = item["labels"]
                if label_delim and isinstance(label, str):
                    label_list = label.split(label_delim)
                else:
                    label_list = [label] if isinstance(label, str) else label
            else:
                label_list = None
                logger.info(f"Data point {idx}: No labels found, setting label_list to None.")
            
            one_hot_labels = item.get("one_hot_labels", None)
            if one_hot_labels is None:
                logger.info(f"Data point {idx}: No one_hot_labels found, setting to None.")

            for chunk in chunks:
                chunk_item = {
                    "input_ids": chunk["input_ids"],
                    "attention_mask": chunk["attention_mask"],
                }
                if "token_type_ids" in chunk and chunk["token_type_ids"] is not None:
                    chunk_item["token_type_ids"] = chunk["token_type_ids"]
                if "labels" in item:
                    chunk_item["labels"] = item["labels"]
                chunk_item["label_list"] = label_list
                chunk_item["one_hot_labels"] = one_hot_labels
                self.items.append(chunk_item)
        logger.info(f"Created FlatChunkDataset with {len(self.items)} chunks.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int | List[int]) -> Dict[str, Any] | List[Dict[str, Any]]:
        """
        Retrieve item(s) by index, supporting both integer and list of indices.
        """
        if isinstance(idx, (list, np.ndarray)):
            return [self.items[i] for i in idx]
        return self.items[idx]

    def __getitems__(self, indices):
        """Handle batched indices from DataLoader"""
        if isinstance(indices, dict):
            # Handle the case where indices is a dictionary
            # This depends on what format indices actually has
            return [self.items[i] for i in range(len(self.items)) if i in indices.values()]
        elif isinstance(indices, (list, np.ndarray)):
            return [self.items[i] for i in indices]
        else:
            return self.items[indices]

    def __repr__(self) -> str:
        num_items = len(self.items)
        keys = self.items[0].keys() if self.items else []
        return f"FlatChunkDataset(num_chunks={num_items}, keys={list(keys)})"


def check_chunk_labels(tokenized_dataset: Dataset, flat_dataset: 'FlatChunkDataset', label_delim: Optional[str] = None) -> bool:
    """
    Sanity check to ensure all chunks of a data point in FlatChunkDataset have the same
    'labels', 'label_list', and 'one_hot_labels' as the original data point.

    Uses the global 'logger' for outputting check results.

    Args:
        tokenized_dataset: Hugging Face Dataset with 'text', 'chunks', 'labels', 'label_list', 'one_hot_labels'.
        flat_dataset: FlatChunkDataset with chunks, 'labels', 'label_list', 'one_hot_labels'.
        label_delim: Optional delimiter used to process labels into label_list.

    Returns:
        bool: True if labels, label_list, and one_hot_labels are consistent for all chunks, False otherwise.
    """
    if "labels" not in tokenized_dataset.column_names:
        logger.info("No 'labels' column in tokenized_dataset. Skipping check.")
        return True

    chunk_counts = [len(item["chunks"]) for item in tokenized_dataset]
    chunk_start = 0
    labels_passed = True
    label_list_passed = True
    one_hot_passed = True

    for idx, (item, chunk_count) in enumerate(zip(tokenized_dataset, chunk_counts)):
        # Expected values from tokenized_dataset
        expected_labels = item["labels"]
        expected_label_list = item.get("label_list", None)
        expected_one_hot = item.get("one_hot_labels", None)
        
        # Process expected label_list for comparison
        if expected_labels is not None:
            if label_delim and isinstance(expected_labels, str):
                computed_label_list = expected_labels.split(label_delim)
            else:
                computed_label_list = [expected_labels] if isinstance(expected_labels, str) else expected_labels
        else:
            computed_label_list = None

        # Get chunk values from flat_dataset
        chunk_labels = [
            flat_dataset[chunk_start + i]["labels"]
            for i in range(chunk_count)
            if "labels" in flat_dataset[chunk_start + i]
        ]
        chunk_label_lists = [
            flat_dataset[chunk_start + i]["label_list"]
            for i in range(chunk_count)
            if "label_list" in flat_dataset[chunk_start + i]
        ]
        chunk_one_hots = [
            flat_dataset[chunk_start + i]["one_hot_labels"]
            for i in range(chunk_count)
            if "one_hot_labels" in flat_dataset[chunk_start + i]
        ]

        if not (chunk_labels or chunk_label_lists or chunk_one_hots):
            logger.info(f"Data point {idx}: No labels, label_list, or one_hot_labels found in chunks. Skipping.")
            chunk_start += chunk_count
            continue

        # Check labels
        for i, chunk_label in enumerate(chunk_labels):
            if isinstance(expected_labels, torch.Tensor) and isinstance(chunk_label, torch.Tensor):
                labels_equal = torch.equal(expected_labels, chunk_label)
            elif isinstance(expected_labels, list) and isinstance(chunk_label, list):
                labels_equal = expected_labels == chunk_label
            else:
                labels_equal = expected_labels == chunk_label
            if not labels_equal:
                logger.info(f"Data point {idx}, chunk {i + 1}: Labels mismatch. Expected {expected_labels}, got {chunk_label}")
                labels_passed = False

        # Check label_list
        for i, chunk_label_list in enumerate(chunk_label_lists):
            if expected_label_list is None and chunk_label_list is None:
                continue
            if expected_label_list is None or chunk_label_list is None:
                logger.info(f"Data point {idx}, chunk {i + 1}: Label_list mismatch. Expected {expected_label_list}, got {chunk_label_list}")
                label_list_passed = False
                continue
            if isinstance(expected_label_list, list) and isinstance(chunk_label_list, list):
                label_list_equal = expected_label_list == chunk_label_list
            else:
                label_list_equal = expected_label_list == chunk_label_list
            if not label_list_equal:
                logger.info(f"Data point {idx}, chunk {i + 1}: Label_list mismatch. Expected {expected_label_list}, got {chunk_label_list}")
                label_list_passed = False

        # Check one_hot_labels
        for i, chunk_one_hot in enumerate(chunk_one_hots):
            if expected_one_hot is None and chunk_one_hot is None:
                continue
            if expected_one_hot is None or chunk_one_hot is None:
                logger.info(f"Data point {idx}, chunk {i + 1}: One_hot_labels mismatch. Expected {expected_one_hot}, got {chunk_one_hot}")
                one_hot_passed = False
                continue
            # Convert to lists for comparison if necessary
            expected_one_hot_list = expected_one_hot if isinstance(expected_one_hot, list) else expected_one_hot.tolist()
            chunk_one_hot_list = chunk_one_hot if isinstance(chunk_one_hot, list) else chunk_one_hot.tolist()
            one_hot_equal = expected_one_hot_list == chunk_one_hot_list
            if not one_hot_equal:
                logger.info(f"Data point {idx}, chunk {i + 1}: One_hot_labels mismatch. Expected {expected_one_hot_list}, got {chunk_one_hot_list}")
                one_hot_passed = False

        chunk_start += chunk_count

    # Log results
    if labels_passed:
        logger.info(f"Labels check passed: All {chunk_start} chunks have consistent labels.")
    else:
        logger.info("Labels check failed: Some chunks have inconsistent labels.")
    if label_list_passed:
        logger.info(f"Label_list check passed: All {chunk_start} chunks have consistent label_list.")
    else:
        logger.info("Label_list check failed: Some chunks have inconsistent label_list.")
    if one_hot_passed:
        logger.info(f"One_hot_labels check passed: All {chunk_start} chunks have consistent one_hot_labels.")
    else:
        logger.info("One_hot_labels check failed: Some chunks have inconsistent one_hot_labels.")

    all_passed = labels_passed and label_list_passed and one_hot_passed
    logger.info(f"Overall check result: {'Passed' if all_passed else 'Failed'}")
    return all_passed


def tokenization_experiment(
    source: Path,
    data: str,
    output_dir: Path,
    label_delim: str = ";",
    max_length: int = 512,
    overlap: int = 50,
):
    """
    Tokenize the 'text' field of a DataFrame using multiple tokenizers and display results.

    Loads the dataset, randomly selects 10 data points, tokenizes the 'text' field with
    multiple tokenizers, displays tokenization results, and saves tokenized data.

    Args:
        source_url (str): Source identifier for data location.
        data (str): Base filename of the CSV data.
        base_dir (str): Base directory for data.
        output_dir (str): Directory to save outputs.
        max_length (int): Maximum token length for tokenization.
        overlap (int): Number of tokens to overlap between chunks.
    """
    # Load DataFrame
    try:
        df = get_data(source, data)
        logger.info(f"Loaded DataFrame with {len(df)} rows.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Randomly select 10 data points
    sample_df = df.sample(n=10, random_state=42).reset_index(drop=True)
    logger.info(f"Selected 10 random data points.")

    # Define tokenizers to test
    tokenizer_configs = [
        {"name": "Bio_ClinicalBERT", "model_id": "emilyalsentzer/Bio_ClinicalBERT"},
        {"name": "PubMedBERT", "model_id": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"},
        {"name": "BERT-base", "model_id": "bert-base-uncased"},
    ]

    # Initialize tokenizers
    tokenizers = {
        config["name"]: AutoTokenizer.from_pretrained(config["model_id"])
        for config in tokenizer_configs
    }

    # Convert sample DataFrame on Hugging Face Dataset for saving
    sample_dataset = Dataset.from_pandas(sample_df)

    # Tokenize and display results for each data point
    for idx, row in sample_df.iterrows():
        text = row["text"]
        logger.info(f"\n┌{'─'*78}┐")
        logger.info(f"│ Data Point {idx+1:<65} │")
        logger.info(f"└{'─'*78}┘")
        logger.info(f"  Original Text (first 200 chars):")
        logger.info(f"    {text[:200]}...")
        
        for tokenizer_name, tokenizer in tokenizers.items():
            logger.info(f"\n  ┌{'─'*74}┐")
            logger.info(f"  │ Tokenizer: {tokenizer_name:<62} │")
            logger.info(f"  └{'─'*74}┘")
            
            # Chunk the text
            chunks = chunk_text(text, tokenizer, max_length, overlap)
            
            # Log results for each chunk
            logger.info(f"    Number of chunks: {len(chunks)}")
            if chunks:
                for chunk_idx, chunk in enumerate(chunks):
                    logger.info(f"      ┌{'─'*70}┐")
                    logger.info(f"      │ Chunk {chunk_idx + 1:<64} │")
                    logger.info(f"      └{'─'*70}┘")
                    
                    input_ids = chunk["input_ids"]
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    
                    logger.info(f"        Tokens (first 50): {tokens[:50]}")
                    logger.info(f"        Input IDs (first 50): {input_ids[:50].tolist()}")
                    
                    # Check for [UNK] tokens
                    unk_count = tokens.count("[UNK]")
                    if unk_count > 0:
                        logger.info(f"        Warning: {unk_count} [UNK] tokens found in chunk {chunk_idx + 1}.")
                    else:
                        logger.info(f"        No [UNK] tokens found in chunk {chunk_idx + 1}.")
            else:
                logger.info(f"    No tokens generated (empty text).")

    # Save tokenized data for each tokenizer (all chunks)
    for tokenizer_name, tokenizer in tokenizers.items():
        def tokenize_function(examples):
            all_chunks = []
            for text in examples["text"]:
                chunks = chunk_text(text, tokenizer, max_length, overlap)
                all_chunks.append(chunks)
            
            return {"chunks": all_chunks}  # Store as list of chunk dicts
        
        tokenized_dataset = sample_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["is_valid"]  # Keep text and chunks
        )

        save_path = os.path.join(output_dir, f"tokenized_{tokenizer_name.lower().replace('_', '-')}")
        tokenized_dataset.save_to_disk(save_path)
        logger.info(f"Saved tokenized dataset (with chunks) for {tokenizer_name} to {save_path}")

    logger.info(f"Tokenization experiment completed!")
    logger.info("\n\n")


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Simplified collate function that directly creates tensors from batch items.
    """
    # Create batched tensors for inputs that should always be present
    batched = {
        "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
    }
    
    # Add token_type_ids if present
    if all("token_type_ids" in item for item in batch):
        batched["token_type_ids"] = torch.tensor([item["token_type_ids"] for item in batch], dtype=torch.long)
    else:
        logger.warning("Some items in batch are missing token_type_ids")
    
    # Add labels if one_hot_labels are present
    if all("one_hot_labels" in item for item in batch):
        batched["labels"] = torch.tensor([item["one_hot_labels"] for item in batch], dtype=torch.float)
    else:
        # Fallback for missing one_hot_labels
        batched["labels"] = torch.zeros((len(batch), 1), dtype=torch.float)
        logger.warning(f"Batch has no valid one_hot_labels, using zero tensor.")
    
    return batched


def print_batch_stats(batch: Dict[str, torch.Tensor], batch_idx: int):
    """
    Print statistics for a batch.
    """
    logger.info(f"Batch {batch_idx}:")
    for key, value in batch.items():
        if key == "labels":
            num_active = value.sum().item()
            avg_labels_per_sample = value.sum(dim=1).mean().item() if value.shape[0] > 0 else 0
            logger.info(f"  {key}: shape={value.shape}, active_labels={num_active:.0f}, avg_labels_per_sample={avg_labels_per_sample:.2f}")
        else:
            logger.info(f"  {key}: shape={value.shape}")
    
    if "attention_mask" in batch:
        non_padding = batch["attention_mask"].sum().item()
        total_tokens = batch["attention_mask"].numel()
        logger.info(f"  Non-padding tokens: {non_padding}/{total_tokens} ({non_padding/total_tokens*100:.2f}%)")


def dummy_training_loop(dataloader: DataLoader, num_labels: int, device: str = "cpu"):
    """
    Simulate a multi-label classification training loop without updating weights.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    ).to(device)
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    logger.info("Starting dummy training loop...")
    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        inputs = {
            key: value.to(device) for key, value in batch.items() if key != "labels"
        }
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        logger.info(f"Batch {batch_idx}: Loss={loss.item():.4f}, Samples={batch_size}")
        print_batch_stats(batch, batch_idx)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    logger.info(f"Dummy training complete. Average loss: {avg_loss:.4f}, Total samples: {total_samples}")


def dummy_pipeline_experiment(
    source: Path,
    data: str,
    output_dir: str,
    max_length: int = 512,
    overlap: int = 50,
    label_delim: Optional[str] = ",",
    batch_size: int = 8
):
    """
    Process dataset, create DataLoader, and run dummy training.

    Args:
        source: Path for data source (e.g., untar_xxx('XURLs.MIMIC4_DEMO')).
        data: Base filename of the CSV (e.g., 'mimic4_icd10_full').
        output_dir: Directory for logs (e.g., 'mimic4_tokenization').
        max_length: Max sequence length for tokenization.
        overlap: Overlap between chunks.
        label_delim: Delimiter for splitting labels.
        batch_size: Number of chunks per batch.
    """
    logger.info('#'*80)
    logger.info("Running Dummy Pipeline")
    logger.info("----------------------")

    # Load data
    df = get_data(source, data)  # Your get_data function
    logger.info(f"Loaded DataFrame with {len(df)} rows.")

    # Sample for testing
    sample_df = df.sample(n=10, random_state=42).reset_index(drop=True)
    sample_dataset = Dataset.from_pandas(sample_df)
    logger.info(f"Created test sample with {len(sample_df)} rows")

    # Process labels
    def process_labels(examples):
        if label_delim:
            labels = [example.split(label_delim) if isinstance(example, str) else example
                      for example in examples['labels']]
        else:
            labels = [[example] if isinstance(example, str) else example
                      for example in examples['labels']]
        return {'label_list': labels}

    sample_dataset = sample_dataset.map(
        process_labels,
        batched=True,
        desc="Processing Labels"
    )

    # One-hot encoding with MultiLabelBinarizer
    mlb_classes = sorted(set(label for labels in sample_dataset['label_list'] if labels for label in labels))
    logger.info(f"Found {len(mlb_classes)} unique labels for MultiLabelBinarizer.")
    mlb = MultiLabelBinarizer(classes=mlb_classes)
    one_hot_labels = mlb.fit_transform(sample_dataset['label_list'])

    def add_one_hot(examples, indices):
        return {'one_hot_labels': [one_hot_labels[idx].tolist() for idx in indices]}

    sample_dataset = sample_dataset.map(
        add_one_hot,
        with_indices=True,
        batched=True,
        desc="Adding one-hot labels"
    )

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    def tokenize_function(examples):
        all_chunks = []
        for text in examples["text"]:
            chunks = chunk_text(text, tokenizer, max_length, overlap)
            all_chunks.append(chunks)
        return {"chunks": all_chunks}

    tokenized_dataset = sample_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["is_valid"]
    )
    # Create FlatChunkDataset
    flat_dataset = FlatChunkDataset(tokenized_dataset, label_delim=label_delim)
    logger.info(f"FlatChunkDataset type: {type(flat_dataset)}")

    # Check labels consistency
    logger.info("Running label consistency check...")
    passed = check_chunk_labels(tokenized_dataset, flat_dataset, label_delim=label_delim)
    logger.info(f"Label check result: {'Passed' if passed else 'Failed'}")

    # Create DataLoader
    dataloader = DataLoader(
        flat_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    logger.info(f"Created DataLoader with batch_size={batch_size}, num_batches={len(dataloader)}")

    # Run dummy training
    num_labels = len(mlb_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_training_loop(dataloader, num_labels, device)

    logger.info(f"Running dummpy pipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize MIMIC4 dataset text field.")
    parser.add_argument("--source_url", type=str, required=True, help="Source identifier for data")
    parser.add_argument("--data", type=str, required=True, help="Base filename of the CSV data")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument(
        "--label_delim",
        type=str,
        default=";",
        help="Delimiter separating labels (default: %(default)s)"
    )
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum length of text chunks (default: 512)')
    parser.add_argument('--overlap', type=int, default=50,
                        help='Overlap between consecutive chunks (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing (default: 8)')
    
    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Data and File Setup
    # -----------------------------------------------------------------------------
    # Resolve source directory 
    source = untar_xxx(eval(args.source_url))
    base_dir = args.base_dir + '/' + args.source_url.split('.')[1]
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------------
    # Logging Setup
    # -----------------------------------------------------------------------------
    logger = setup_logger(output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    tokenization_experiment(
        source=source,
        data=args.data,
        output_dir=output_dir,
        label_delim=args.label_delim,
        max_length=args.max_length,
        overlap=args.overlap,
    )

    dummy_pipeline_experiment(
        source=source,
        data=args.data,
        output_dir=output_dir,
        max_length=args.max_length,
        overlap=args.overlap,
        label_delim=args.label_delim,  # Adjustable
        batch_size=args.batch_size      # Adjustable
    )