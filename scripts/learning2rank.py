# =============================================================================
# Imports
# =============================================================================
# Standard library imports
import argparse
import logging
import os
import sys
import warnings
import json
from tabulate import tabulate
from datetime import datetime
from pathlib import Path
import time
import math
from datetime import timedelta
from colorama import Fore, Style, init
import matplotlib.pyplot as plt

# Initialize colorama for cross-platform colored output
init()

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file
from torchmetrics.functional import retrieval_normalized_dcg, retrieval_precision
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login, whoami
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    get_scheduler,
    pipeline,
    PreTrainedModel,
    PretrainedConfig
)

from sklearn.preprocessing import MultiLabelBinarizer

# Local application imports
from xcube.data.all import *
from xcube.imports import *
from xcube.l2r.models.core import LTRConfig, LTRModel
from tqdm import tqdm

from flash_attn_bf16_guard import install_bf16_boundary_guard
install_bf16_boundary_guard()


# =============================================================================
# Constants
# =============================================================================
global base_dir
base_dir = None
# Set a fixed random seed for reproducibility
random_seed = 42


# =============================================================================
# Warnings and Configuration
# =============================================================================
warnings.filterwarnings(action="ignore")
torch.backends.cudnn.benchmark = True


# =============================================================================
# Helper Functions
# =============================================================================
def log_section(message, char="=", length=80):
    """
    Creates a visually appealing log message with a box around it.

    Args:
        message (str): The message to display
        char (str): Character to use for the box (default: "=")
        length (int): Width of the box (default: 80)
    """
    border = char * length
    padding = (length - len(message) - 4) // 2
    left_padding = " " * padding
    right_padding = " " * (length - len(message) - 4 - padding)

    # Choose emoji based on border character
    section_emoji = "📌"
    if char == "=":
        section_emoji = "📌"
    elif char == "*":
        section_emoji = "🌟"
    elif char == "+":
        section_emoji = "✨"
    elif char == "-":
        section_emoji = "📋"
    elif char == "!":
        section_emoji = "⚠️"

    logging.info(border)
    logging.info(
        f"{char}{left_padding} {section_emoji} {message} {right_padding}{char}"
    )
    logging.info(border)


def log_print_output(func, *args, prefix="📊", **kwargs):
    """
    Capture print output from any function and log it

    Args:
        func: The function to call
        *args: Positional arguments to pass to the function
        prefix: Emoji/prefix for the log message
        **kwargs: Keyword arguments to pass to the function

    # Usage examples:
    log_print_output(model.print_trainable_parameters)
    log_print_output(print, "Hello World", prefix="🔍")
    log_print_output(some_function, arg1, arg2, keyword_arg=value, prefix="⚡")
    """
    with io.StringIO() as buf, redirect_stdout(buf):
        func(*args, **kwargs)
        output = buf.getvalue().strip()
        if output:  # Only log if there's actual output
            logging.info(f"{prefix} {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetuning Large Language Model")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output path for the finetuned LLM model.",
    )
    parser.add_argument(
        "--source_url", type=str, default="XURLs.MIMIC4", help="Source url"
    )
    parser.add_argument(
        "--data", type=str, default="mimic4_icd10_full", help="Filename of the raw data"
    )
    parser.add_argument(
        "--data_l2r_fname_prefix",
        type=str,
        required=True,
        help="Data source identifier (e.g., mimic4_icd10_tok_lbl)",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="training_l2r_log",
        help="Filename to write logs to",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Path to the base directory that will contain intermediate results and models.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="your_username/mistral-7b-medical-lora",
        help="Hugging Face hub model id (replace with your username/model name)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Model name (e.g., mistralai/Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length (default: 512). Adjust as needed (e.g., 1024 if your summaries are longer and 48GB VRAM allows)",
    )
    parser.add_argument(
        "--do_fresh_training",
        action="store_true",
        default=False,
        help="If set, perform fresh training (default: False)",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        action="store_true",
        default=False,
        help="Load model and tokenizer from Hub before training",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for the optimizer (default: 2e-4)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for training (default: 500)",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="ndcg@10",
        choices=["ndcg", "ndcg@25"],
        help="Metric to determine the best model (default: ndcg@k).",
    )
    parser.add_argument(
        "--l2r_boot_dir",
        type=str,
        default="mimic4_l2rboot",
        help="Directory for l2r boot files (default: mimic4_l2rboot)",
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate a report (default: False)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device during training (default: 1)",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size per device during evaluation (default: 1)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--skip_push_to_hub",
        action="store_true",
        default=False,
        help="If set, skip pushing model/tokenizer checkpoints to the Hugging Face Hub (default: False)",
    )
    parser.add_argument(
        "--label_space",
        type=str,
        default="label",
        choices=["label", "candidate"],
        help="'label' (default) trains against every label in the --data_l2r_fname_prefix data "
        "as usual -- this is also what a cluster-router run uses, just pointed at "
        "aggregate_cluster_mig.py's combined cluster-level data via --data_l2r_fname_prefix, no "
        "extra flag needed there (see README_AMAZON3M.md's full-scale candidate-generator plan). "
        "'candidate' restricts each document's forward pass/loss to a bounded per-document "
        "candidate set (true positives + same-cluster hard negatives) instead of scoring every "
        "label -- requires cluster_assignments.json/cluster_to_labels.json "
        "(scripts/build_label_clusters.py) and labels_partition.json under --base_dir, and a "
        "plain (non-cluster) Stage 3 bootstrap run so label ids line up with the clustering.",
    )
    parser.add_argument(
        "--candidate_budget",
        type=int,
        default=256,
        help="Fixed number of (true positive + same-cluster hard negative) label ids scored per "
        "document when --label_space=candidate (default: 256).",
    )
    parser.add_argument(
        "--label_embed_rank",
        type=int,
        default=None,
        help="If set, factorizes LTRModel's per-label embedding table into a low-rank "
        "(num_labels, label_embed_rank) base plus one shared up-projection to hidden_size, "
        "instead of a full-rank (num_labels, hidden_size) table -- needed at Amazon-3M's full "
        "~2.81M-label scale (see README_AMAZON3M.md's full-scale candidate-generator plan).",
    )

    return parser.parse_args()


def setup_logger(**kwargs):
    """
    Configure logging for the application.

    Sets up both console and file logging based on the quiet flag in args.
    Logs include detailed information such as function name, line number,
    and timestamp. Log files are saved in a 'logs' directory within base_dir
    with a timestamp in their filename.
    """
    # Access the 'base_dir' and 'logfile' arguments from kwargs.
    quiet = kwargs.get("quiet")
    logfile = kwargs.get("logfile")
    base_dir = kwargs.get("base_dir")

    # Configure the root logger based on the quiet flag
    if quiet:
        logging.basicConfig(
            level=logging.CRITICAL
        )  # Suppresses all logging below CRITICAL
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d:%(funcName)s]",
        )

    # Set up file logging globally
    log_base_dir = Path(base_dir) / "logs"
    log_base_dir.mkdir(exist_ok=True, parents=True)

    # Format the current date and time for the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = (
        f"{logfile}_{current_time}.log"
        if logfile
        else f"default_logfile_{current_time}.log"
    )
    log_filepath = log_base_dir / log_filename

    # Create file handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)  # Set logging level for the file handler

    # Create formatter and add it to the file handler
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d:%(funcName)s]"
    )
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    logging.info(f"📝 Logging initialized. Log file created at: {log_filepath}")

    return log_filepath


def print_args(**kwargs):
    """
    Logs and prints the provided keyword arguments in a formatted manner.

    Args:
        **kwargs: Arbitrary keyword arguments representing command-line parameters.
    """
    # Create a formatted string of key-value pairs.
    formatted_args = "\n".join(f"  🔹 {key}: {value}" for key, value in kwargs.items())

    # Log and print the formatted arguments.
    logging.info("🛠️ Command-line Arguments:")
    logging.info("\n" + formatted_args)
    logging.info("➖" * 30)


def authenticate_hf(token):
    """Authenticate with Hugging Face using a token."""
    login(token=token)
    logging.info("🔑 Successfully authenticated with Hugging Face Hub")


def get_data(source, l2r_boot_dir, data, data_l2r_fname_prefix):
    """
    Load CSV data and L2R data into pandas DataFrames.

    Constructs the file paths using the provided parameters,
    reads the CSV file and L2R feather file, and ensures that
    the 'text' and 'labels' columns are strings.

    Args:
        source (str): The source directory or identifier used to locate the CSV file.
        l2r_boot_dir (str): Directory containing the L2R feather file.
        data (str): The base filename (without extension) of the CSV data.
        data_l2r (str): The base filename (without extension) of the L2R feather data.

    Returns:
        tuple: (df_data, df_l2r) containing the main data and L2R data DataFrames.
    """
    # Construct the file path with a '.csv' extension for the main data.
    filepath = join_path_file(data, source, ext=".csv")
    logging.info(f"📂 Loading main data from: {filepath}")

    # Read the CSV file, selecting only the necessary columns,
    # and explicitly define their data types.
    df_data = pd.read_csv(
        filepath,
        header=0,
        usecols=["text", "labels", "is_valid"],
        dtype={"text": str, "labels": str, "is_valid": bool},
    )

    # Convert the selected columns to string type (if needed).
    df_data[["text", "labels"]] = df_data[["text", "labels"]].astype(str)
    logging.info(f"✅ Successfully loaded main data: {len(df_data)} rows")

    # Construct the file path for the L2R feather files
    l2r_filepath = os.path.join(
        l2r_boot_dir, f"{data_l2r_fname_prefix}_tok_rank_per_lbl.ft"
    )
    logging.info(f"📂 Loading L2R data from: {l2r_filepath}")

    tokens_filepath = os.path.join(l2r_boot_dir, f"{data_l2r_fname_prefix}_tok.ft")
    logging.info(f"📂 Loading L2R tokens from: {tokens_filepath}")

    labels_filepath = os.path.join(l2r_boot_dir, f"{data_l2r_fname_prefix}_lbl.ft")
    logging.info(f"📂 Loading L2R labels from: {labels_filepath}")

    try:
        # Read the feather files for L2R data
        df_l2r = pd.read_feather(l2r_filepath)
        df_toks = pd.read_feather(tokens_filepath)
        df_lbs = pd.read_feather(labels_filepath)
        logging.info(f"✅ Successfully loaded L2R data: {len(df_l2r)} rows")
        logging.info(f"✅ Successfully loaded L2R tokens: {len(df_toks)} tokens")
        logging.info(f"✅ Successfully loaded L2R labels: {len(df_lbs)} rows")
    except Exception as e:
        logging.error(f"❌ Failed to load L2R data: {e}")
        logging.error(f"❌ Failed to load L2R tokens: {e}")
        logging.error(f"❌ Failed to load L2R labels: {e}")
        raise RuntimeError(f"Failed to load L2R data from {l2r_boot_dir}: {e}")

    logging.info(
        f"🔄 Total data loaded: {len(df_data)} main rows, {len(df_l2r)} L2R rows"
    )

    return df_data, df_l2r, df_toks, df_lbs


def load_datasets(source, l2r_boot_dir, data, data_l2r_fname_prefix):
    """
    Load main and L2R datasets with comprehensive logging.

    Args:
        source (str): Source directory for the main dataset
        l2r_boot_dir (str): Directory containing the L2R data
        data (str): Filename for the main dataset
        data_l2r (str): Filename for the L2R dataset

    Returns:
        tuple: (df_data, df_l2r) containing the main data and L2R data DataFrames
    """
    try:
        df_data, df_l2r, df_toks, df_lbs = get_data(
            source, l2r_boot_dir, data, data_l2r_fname_prefix
        )
        logging.info(f"✨ Successfully loaded both datasets:")
        logging.info(f"   - 📄 Main dataset: {len(df_data)} records")
        logging.info(f"   - 📄 L2R dataset: {len(df_l2r)} records")
        logging.info(f"   - 🔤 Tokens: {len(df_toks)} items")
        logging.info(f"   - 🏷️ Labels: {len(df_lbs)} items")

        # Log data sample for verification
        logging.debug("🔍 Sample of main dataset (first 2 rows):")
        logging.debug(f"\n{df_data.head(2)}")
        logging.debug("🔍 Sample of L2R dataset (first 2 rows):")
        logging.debug(f"\n{df_l2r.head(2)}")

        logging.info("✅ Data loading completed successfully")
    except Exception as e:
        logging.error(f"❌ Failed to load datasets: {e}")
        raise RuntimeError(f"Failed to load datasets: {e}")

    return df_data, df_l2r, df_toks, df_lbs


def check_model_existence(output_dir, task_hub_model_id, hf_token=None):
    """
    Check if a model exists locally and on Hugging Face Hub.

    Args:
        output_dir (str): Directory where the model would be saved locally
        task_hub_model_id (str): Hugging Face model ID to check
        hf_token (str, optional): Hugging Face authentication token

    Returns:
        tuple: (model_exists_locally, model_exists_on_hub)
    """
    logging.info("🔍 Checking model existence locally and on Hugging Face Hub...")

    # Check local existence
    model_exists_locally = os.path.exists(os.path.join(output_dir, "config.json"))
    if model_exists_locally:
        logging.info(f"✅ Model exists locally at: {output_dir}")
    else:
        logging.info(f"❌ Model not found locally at: {output_dir}")

    # Check Hub existence
    model_exists_on_hub = False
    api = HfApi()
    try:
        model_exists_on_hub = (
            api.model_info(task_hub_model_id, token=hf_token).id is not None
            if hf_token
            else False
        )
        if model_exists_on_hub:
            logging.info(
                f"✅ Model exists on Hugging Face Hub with ID: {task_hub_model_id}"
            )
        else:
            logging.info(
                f"❌ Model not found on Hugging Face Hub with ID: {task_hub_model_id}"
            )
    except Exception as e:
        # Check if it's a 404 error (Repository Not Found)
        if "404" in str(e) and "Repository Not Found" in str(e):
            logging.info(
                f"❌ Model repository does not exist on Hugging Face Hub: {task_hub_model_id}"
            )
            logging.debug(f"🔍 Detail: {e}")
        else:
            # For other API errors
            logging.warning(
                f"❓ Failed to check model on Hub due to API error: {type(e).__name__}"
            )
            logging.warning(f"🔍 Error details: {e}")

        # Set the flag to false regardless of error type
        model_exists_on_hub = False

    # Summarize model existence status
    if model_exists_locally or model_exists_on_hub:
        logging.info("📁 Model exists either locally or on Hub")
    else:
        logging.info("🔄 Model doesn't exist yet - will need to train from scratch")

    return model_exists_locally, model_exists_on_hub


def quantize_l2r_data(
    df_l2r: "pd.DataFrame",
    qnts: Optional[torch.Tensor] = None,
    device: Optional[str] = None,
):
    """
    Quantizes the rank column of a token-to-rank (containing ranks of all tokens per label) DataFrame by mapping ranks into discrete bins based on provided quantiles.
    This process is useful for transforming continuous rank values into categorical relevance scores, often used in
    ranking or recommendation systems.

    Args:
        df_l2r (pd.DataFrame): Input pandas DataFrame containing ranking data. Must include columns:
            - 'token': Identifier for items (e.g., item IDs).
            - 'label': Categorical label associated with each token (e.g., category or group ID).
            - 'rank': Continuous rank or score of tokens per label to be quantized (e.g., relevance score or position).
            - 'bcx_mutual_info': Optional mutual information metric (not used in quantization but present in input).
        qnts (torch.Tensor, optional): 1D tensor of quantile values (between 0 and 1) used to define bin edges for
            quantization. If None, a default set of 101 quantiles is generated, starting at 0 and using a logarithmic
            scale from 1e-4 to 1.0.
        device (str, optional): Target device for tensor computations ('cpu' or 'cuda'). If None, uses CPU.

    Returns:
        torch.Tensor: A 3D tensor of shape (num_labels, num_tokens, 4), where:
            - num_labels: Number of unique labels in the input DataFrame.
            - num_tokens: Number of tokens per label.
            - Last dimension contains [token, label, rank, relevance_score], where relevance_score is the quantized
              rank based on the provided quantiles.

    Raises:
        ValueError: If the input DataFrame lacks required columns ('token', 'label', 'rank') or if the data tensor
            does not have exactly 3 columns after conversion.

    Example:
        Input df_l2r.head():
            token  label  bcx_mutual_info     rank
            0      0      -6.238006        13380.0
            1      0      -5.998233         9470.0
            2      0      -6.463400        12812.0
            ...

        Output tensor shape: (num_labels, num_tokens, 4)
        Each entry: [token_id, label_id, original_rank, quantized_relevance_score]
    """
    # Log the start of the quantization process with input details
    num_rows = len(df_l2r)
    num_unique_labels = len(df_l2r["label"].unique())
    num_unique_tokens = len(df_l2r["token"].unique())
    logging.info(
        f"⏳ Starting quantization of ranks for DataFrame with {num_rows} rows. Containing {num_unique_tokens} unique tokens & {num_unique_labels} unique labels"
    )

    # Validate that the input DataFrame contains all required columns
    required_columns = ["token", "label", "rank"]
    if not all(col in df_l2r.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # If quantiles are not provided, generate a default set of 101 quantiles
    # Starting with 0 and using a logarithmic scale from 1e-4 to 1 for smooth bin distribution
    if qnts is None:
        qnts = torch.cat(
            [
                torch.tensor([0.0]),  # Include 0 as the first quantile
                torch.logspace(
                    start=torch.log10(torch.tensor(1e-4)),  # Start at 1e-4
                    end=torch.log10(torch.tensor(1.0)),  # End at 1.0
                    steps=100,  # Generate 100 steps
                ),
            ]
        )

    # Log the number of quantization levels (number of quantiles)
    num_quantiles = len(qnts)
    logging.info(
        f"🔄 Quantizing those {num_unique_tokens} unique token ranks into {num_quantiles} quantization levels for each label"
    )

    # Move the quantiles tensor to the specified device (e.g., 'cuda' or 'cpu')
    qnts = to_device(qnts, device)

    # Convert the relevant DataFrame columns (token, label, rank) to a PyTorch tensor
    # Only select the columns needed for processing to optimize memory usage
    data = torch.tensor(
        df_l2r[required_columns].to_numpy(), device=device, dtype=torch.float
    )

    # Verify that the tensor has the expected shape: (num_rows, 3)
    if data.shape[1] != 3:
        raise ValueError(
            f"Expected 3 columns (token, label, rank), got {data.shape[1]}"
        )

    # Sort the data by the 'label' column (index 1) to group tokens by label
    # This ensures that all tokens with the same label are contiguous
    _, sort_indices = data[:, 1].sort()
    data = data[sort_indices]

    # Use np.unique to get unique labels and their starting indices in the sorted tensor
    # Convert the label column to NumPy for np.unique, which supports return_index
    labels, split_indices = np.unique(data[:, 1].cpu().numpy(), return_index=True)
    split_indices = split_indices[1:]  # Exclude the first index (0) for splitting

    # Split the data into sub-arrays, each containing tokens for a specific label
    # Convert to NumPy temporarily for splitting, as NumPy's split is efficient here
    data_splits = np.split(data.cpu().numpy(), split_indices)

    # Stack the split arrays into a 3D tensor: (num_labels, num_tokens_per_label, 3)
    # This organizes the data by label, with each label having its own token set
    data = torch.tensor(np.stack(data_splits), device=device)

    # Compute quantile bins for the ranks (last column) across all tokens for each label
    # bins shape: (num_quantiles, num_labels), where each column defines bin edges for a label
    bins = torch.quantile(data[:, :, -1], qnts, dim=1)

    # Compute relevance scores by finding which bin each rank falls into
    # searchsorted finds the insertion point for each rank in the sorted bins
    # Subtract from the number of bins to get a relevance score (higher rank -> higher score)
    relv_scores = bins.shape[0] - torch.searchsorted(
        bins.T, data[:, :, -1], right=False
    )

    # Append the relevance scores as a new column to the data tensor
    # New shape: (num_labels, num_tokens, 4), where the last dimension is [token, label, rank, score]
    data = torch.cat((data, relv_scores.unsqueeze(-1)), dim=-1)

    # Log the completion of the quantization process with output details
    output_shape = data.shape
    logging.info(
        f"✅ Completed quantization: Produced tensor of shape {output_shape} "
        f"with {num_quantiles} quantization levels per label"
    )

    return data


def to_device(tensor: torch.Tensor, device: Optional[str]):
    """
    Moves a tensor to the specified device (e.g., 'cpu' or 'cuda').

    Args:
        tensor (torch.Tensor): Input tensor to move.
        device (str, optional): Target device. If None, returns the tensor unchanged.

    Returns:
        torch.Tensor: Tensor on the specified device.
    """
    return tensor.to(device) if device else tensor


def test_scored_tokens(
    scored_tokens: torch.Tensor,
    df_toks: pd.DataFrame,
    df_lbs: pd.DataFrame,
    num_labels_to_sample: int = 3,
    num_tokens_per_label: int = 100,
    min_top_relevant: int = 50,
):
    """
    Tests the scored tokens by sampling tokens for randomly selected labels, ensuring a mix of relevance scores.
    Attempts to sample min_top_relevant tokens from the top relevance category, filling the remaining up to
    num_tokens_per_label with tokens in descending order of relevance scores. Displays the results in a formatted
    table using logging.debug, including an index, token values, label values, ranks, and quantized relevance scores,
    sorted by relevance score in descending order.

    Args:
        scored_tokens (torch.Tensor): Tensor of shape (num_labels, num_tokens, 4) containing
            [token, label, rank, relevance_score] for each label-token pair, as output by quantize_l2r_data.
        df_toks (pd.DataFrame): DataFrame with columns ['token', 'tok_val'] mapping token IDs to token values.
        df_lbs (pd.DataFrame): DataFrame with columns ['lbl', 'lbl_val'] mapping label IDs to label values.
        num_labels_to_sample (int): Number of labels to randomly select for testing. Default is 3.
        num_tokens_per_label (int): Number of tokens to sample per label. Default is 100.
        min_top_relevant (int): Minimum number of tokens to sample from the top relevance category. Default is 50.

    Raises:
        ValueError: If scored_tokens has incorrect shape, or if df_toks/df_lbs lack required columns.

    Example Output:
        For each selected label, a table is logged with columns:
        Index | Token ID | Token Value | Label ID | Label Value | Rank       | Relevance Score
        ------|---------|-------------|----------|-------------|------------|---------------
        1     | 0       | [PAD]       | 0        | 00160J6     | 13380.0    | 101
        ...
    """
    # Validate inputs
    if scored_tokens.shape[2] != 4:
        raise ValueError(
            f"Expected scored_tokens with shape (num_labels, num_tokens, 4), got {scored_tokens.shape}"
        )
    if not all(col in df_toks.columns for col in ["token", "tok_val"]):
        raise ValueError("df_toks must contain columns: ['token', 'tok_val']")
    if not all(col in df_lbs.columns for col in ["lbl", "lbl_val"]):
        raise ValueError("df_lbs must contain columns: ['lbl', 'lbl_val']")

    num_labels, num_tokens, _ = scored_tokens.shape

    # Ensure num_labels_to_sample is not greater than available labels
    num_labels_to_sample = min(num_labels_to_sample, num_labels)
    if num_labels_to_sample <= 0:
        raise ValueError("No labels available to sample")

    # Randomly select labels
    selected_label_indices = np.random.choice(
        num_labels, size=num_labels_to_sample, replace=False
    )
    logging.debug(
        f"Selected {num_labels_to_sample} labels for testing: {selected_label_indices.tolist()}"
    )

    # Create dictionaries for token and label lookups
    tok_dict = dict(zip(df_toks["token"], df_toks["tok_val"]))
    lbl_dict = dict(zip(df_lbs["lbl"], df_lbs["lbl_val"]))

    for label_idx in selected_label_indices:
        # Get data for the current label
        label_data = scored_tokens[label_idx]  # Shape: (num_tokens, 4)
        token_ids = label_data[:, 0].long()
        label_ids = label_data[:, 1].long()
        ranks = label_data[:, 2].float()
        relevance_scores = label_data[:, 3].long()

        # Check if enough tokens are available
        if num_tokens < num_tokens_per_label:
            logging.warning(
                f"Label {label_idx}: Only {num_tokens} tokens available (requested {num_tokens_per_label})"
            )
            num_tokens_per_label_actual = num_tokens
        else:
            num_tokens_per_label_actual = num_tokens_per_label

        # Identify top relevance score (assumed to be the maximum)
        max_relevance = relevance_scores.max().item()
        logging.debug(
            f"Label {label_idx} (ID: {label_ids[0].item()}): Maximum relevance score is {max_relevance}"
        )

        # Get indices of tokens with top relevance score
        top_relevant_indices = (relevance_scores == max_relevance).nonzero(
            as_tuple=True
        )[0]
        num_top_available = len(top_relevant_indices)
        num_top_to_sample = min(min_top_relevant, num_top_available)

        if num_top_available < min_top_relevant:
            logging.warning(
                f"Label {label_idx}: Only {num_top_available} tokens with top relevance score "
                f"(need {min_top_relevant})"
            )

        # Sample top relevance tokens
        top_sampled_indices = (
            np.random.choice(
                top_relevant_indices.cpu().numpy(),
                size=num_top_to_sample,
                replace=False,
            )
            if num_top_to_sample > 0
            else np.array([])
        )

        # Sample remaining tokens in descending order of relevance scores
        num_remaining = num_tokens_per_label_actual - num_top_to_sample
        if num_remaining > 0:
            # Get indices of non-top relevance tokens
            non_top_indices = (relevance_scores < max_relevance).nonzero(as_tuple=True)[
                0
            ]
            if len(non_top_indices) < num_remaining:
                logging.warning(
                    f"Label {label_idx}: Only {len(non_top_indices)} non-top relevance tokens "
                    f"available (need {num_remaining})"
                )
                num_remaining = min(num_remaining, len(non_top_indices))

            if num_remaining > 0:
                # Sort non-top tokens by relevance score (descending)
                non_top_scores = relevance_scores[non_top_indices]
                _, sort_indices = non_top_scores.sort(descending=True)
                remaining_indices = (
                    non_top_indices[sort_indices][:num_remaining].cpu().numpy()
                )
            else:
                remaining_indices = np.array([])
        else:
            remaining_indices = np.array([])

        # Combine all sampled indices
        sampled_indices = np.concatenate([top_sampled_indices, remaining_indices])
        if len(sampled_indices) < num_tokens_per_label_actual:
            logging.warning(
                f"Label {label_idx}: Sampled only {len(sampled_indices)} tokens "
                f"(requested {num_tokens_per_label_actual})"
            )

        # Get data for sampled tokens
        sampled_data = label_data[sampled_indices]

        # Sort by relevance score in descending order (for display)
        _, sort_indices = sampled_data[:, 3].sort(descending=True)
        sampled_data = sampled_data[sort_indices]

        # Extract sorted data
        sampled_tokens = sampled_data[:, 0].long().cpu().numpy()
        sampled_labels = sampled_data[:, 1].long().cpu().numpy()
        sampled_ranks = sampled_data[:, 2].float().cpu().numpy()
        sampled_scores = sampled_data[:, 3].long().cpu().numpy()

        # Create table for pretty printing with index column
        table_data = []
        for idx, (t, l, r, s) in enumerate(
            zip(sampled_tokens, sampled_labels, sampled_ranks, sampled_scores), 1
        ):
            token_val = tok_dict.get(t, "Unknown")
            label_val = lbl_dict.get(l, "Unknown")
            table_data.append([idx, t, token_val, l, label_val, r, s])

        # Format table using tabulate
        headers = [
            "Index",
            "Token ID",
            "Token Value",
            "Label ID",
            "Label Value",
            "Rank",
            "Relevance Score",
        ]
        table = tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".1f")

        # Log the table
        logging.debug(
            f"\nSampled tokens for Label {label_idx} (ID: {sampled_labels[0]}, Value: {lbl_dict.get(sampled_labels[0], 'Unknown')}):\n{table}\n"
        )


def load_base_model_and_tokenizer_old(model_name, hub_model_id, hf_token):
    """
    Load the base model and tokenizer with proper error handling.

    Args:
        model_name (str): Name of the base model (original foundation model)
        hub_model_id (str): Hugging Face model ID for the PEFT model
        hf_token (str): Hugging Face API token for authentication

    Returns:
        tuple: (peft_model, tokenizer) or (None, None) if loading fails
    """
    try:
        # Input validation
        if not all([model_name, hub_model_id, hf_token]):
            logging.error(
                "❌ Missing required parameters - all parameters must be provided"
            )
            return None, None

        # Initialize the Hugging Face API
        api = HfApi()

        # Get repository info with error handling
        try:
            logging.info(
                f"🔍 Fetching repository information for PEFT model: {hub_model_id}"
            )
            repo_info = api.repo_info(repo_id=hub_model_id, token=hf_token)
        except Exception as e:
            logging.error(
                f"❌ Failed to retrieve repository information for PEFT model: {str(e)}"
            )
            return None, None

        # Extract and log the last modified timestamp
        if hasattr(repo_info, "last_modified") and repo_info.last_modified:
            last_modified_str = repo_info.last_modified.strftime("%Y-%m-%d %H:%M:%S")
            logging.info(
                f"📅 PEFT model '{hub_model_id}' was last saved on: {last_modified_str}"
            )
        else:
            logging.warning(
                f"⚠️ Could not retrieve last saved timestamp for PEFT model '{hub_model_id}'"
            )

        # Load the base model with error handling
        try:
            logging.info(f"🏛️ Loading BASE MODEL: '{model_name}' (foundation model)")
            # base_model = AutoModelForCausalLM.from_pretrained(
            #     model_name,
            #     device_map="auto",
            #     trust_remote_code=True
            # )
            base_model = AutoModel.from_pretrained(
                model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                device_map="auto",
            )
            logging.info(f"✅ Successfully loaded BASE MODEL: '{model_name}'")
        except Exception as e:
            logging.error(f"❌ Failed to load BASE MODEL '{model_name}': {str(e)}")
            return None, None

        # Load the PEFT model with error handling
        try:
            logging.info(
                f"🔌 Loading PEFT MODEL: '{hub_model_id}' (adapter on top of base model)"
            )
            # model = PeftModel.from_pretrained(base_model, hub_model_id, token=hf_token)
            logging.info(f"✅ Successfully loaded PEFT MODEL: '{hub_model_id}'")
        except Exception as e:
            logging.error(f"❌ Failed to load PEFT MODEL '{hub_model_id}': {str(e)}")
            return None, None

        # Load the tokenizer with error handling
        try:
            logging.info(f"🔤 Loading tokenizer from PEFT model: '{hub_model_id}'")
            # tokenizer = AutoTokenizer.from_pretrained(hub_model_id, token=hf_token)
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", token=hf_token
            )

            # Ensure pad token is set
            if tokenizer.pad_token is None:
                logging.info("🔄 Setting pad_token to eos_token for tokenizer")
                tokenizer.pad_token = tokenizer.eos_token

            logging.info(f"✅ Successfully loaded tokenizer from '{hub_model_id}'")
        except Exception as e:
            logging.error(
                f"❌ Failed to load tokenizer from '{hub_model_id}': {str(e)}"
            )
            return None, None

        logging.info(
            "🎉 COMPLETE: Successfully loaded base model, PEFT model, and tokenizer"
        )
        model = base_model
        return model, tokenizer

    except Exception as e:
        logging.error(f"💥 Unexpected error in load_base_model_and_tokenizer: {str(e)}")
        return None, None


def load_base_model_and_tokenizer(
    model_name="mistralai/Mistral-7B-Instruct-v0.3", lora_config=None
):
    """
    Load a pre-trained model and tokenizer, apply 4-bit quantization and LoRA.

    Args:
        model_name (str): Name of the model to load from Hugging Face.
        lora_config (LoraConfig, optional): Custom LoRA configuration. If None, default is used.

    Returns:
        tuple: (model, tokenizer) - The quantized model with LoRA and the tokenizer.

    Raises:
        ValueError: If model_name is invalid.
        RuntimeError: If model or tokenizer loading fails or LoRA modules are incompatible.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string")

    logging.info("🚀 Starting model and tokenizer loading process...")

    # Define 4-bit quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    logging.info(f"📊 Quantization config: {quantization_config}")

    # Load tokenizer
    try:
        logging.info(f"🔤 Loading tokenizer for model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            logging.info("📝 Setting pad token to eos token...")
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise RuntimeError(f"Tokenizer loading failed: {e}")

    # Determine model type and load appropriately
    try:
        logging.info("🧠 Loading base model with quantization...")
        # torch_dtype=bfloat16 below only governs *parameter* dtype. Buffers
        # built at construction time from torch.get_default_dtype() (e.g.
        # Phi-3-small's RotaryEmbedding cos/sin cache in its custom
        # positional_embedding.py, which passes dtype=None -> get_default_
        # dtype()) are NOT covered by torch_dtype and default to fp32.
        # apply_rotary_pos_emb then upcasts the bf16 query/key to fp32 via
        # that cache, and the fp32 key gets packed into kv -- which is what
        # actually reaches flash-attn's fp16/bf16-only kernel, not the
        # attention output itself. Confirmed via a flash_attn_varlen_kvpacked_
        # func hook (both q and kv arrived fp32) and via a verified fix
        # experiment (casting q/kv back to bf16 immediately before the kernel
        # let training proceed). Setting the process default dtype for the
        # duration of construction makes the cache bf16 from the start;
        # restored in `finally` so it doesn't leak into unrelated code.
        prev_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            if "bert" in model_name.lower():
                # BERT-based models use AutoModelForMaskedLM
                model = AutoModel.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # Default to causal LM for models like Mistral
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,  # bnb_4bit_compute_dtype above only
                    # governs the quantized linear layers; non-quantized submodules
                    # (attention/hidden states) still default to fp32 without this,
                    # which breaks custom fp16/bf16-only attention kernels (e.g.
                    # Phi-3-small's block-sparse/flash-attn: "FlashAttention only
                    # support fp16 and bf16 data type"). Harmless for Mistral/Llama.
                )
        finally:
            torch.set_default_dtype(prev_default_dtype)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

    # Define default LoRA configuration based on model type
    if lora_config is None:
        logging.info("🔧 Setting up default LoRA configuration...")
        if "bert" in model_name.lower():
            # LoRA config for BERT-based models
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "key", "value", "output.dense"],
                lora_dropout=0.05,
                bias="none",
                task_type="MASKED_LM",  # Use MASKED_LM for BERT
            )
        else:
            # LoRA config for causal LM models
            # Phi-3-mini/medium fuse q/k/v into qkv_proj; Phi-3-small (loaded
            # via trust_remote_code) uses a differently-named fused
            # query_key_value/dense pair. Both differ from Llama/Mistral's
            # separate q_proj/k_proj/v_proj/o_proj.
            model_type = getattr(model.config, "model_type", "").lower()
            if model_type == "phi3":
                target_modules = ["qkv_proj", "o_proj"]
            elif model_type == "phi3small":
                target_modules = ["query_key_value", "dense"]
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

    # Log LoRA configuration
    logging.info(
        f"🔍 LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
        f"targets={lora_config.target_modules}, dropout={lora_config.lora_dropout}"
    )

    # Apply LoRA
    try:
        logging.info("🧩 Applying LoRA adapters to model...")
        model = get_peft_model(model, lora_config)
        # PEFT initializes new LoRA A/B weights in fp32 regardless of the base
        # model's dtype (no dtype= on LoraConfig, no prepare_model_for_kbit_
        # training call) -- confirmed empirically against peft==0.20.0. Cast
        # back to bf16 to match the bf16 base/bnb compute dtype; verified this
        # doesn't touch bitsandbytes' Linear4bit quantized weights (stored as
        # uint8, protected from dtype casts) -- only the LoRA layers move.
        model = model.to(torch.bfloat16)
        log_print_output(func=model.print_trainable_parameters)
    except Exception as e:
        logging.error(f"Failed to apply LoRA: {e}")
        raise RuntimeError(f"LoRA application failed: {e}")

    logging.info("✅ Model and tokenizer successfully loaded!")
    return model, tokenizer


def chunk_text_new(
    text: str,
    is_valid: bool,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    overlap: int = 50,
    device: str = "cpu",
) -> List[Dict]:
    """
    Split text into chunks that fit within max_length tokens, with optional overlap.

    Args:
        text (str): Input text to chunk.
        is_valid (bool): Flag to include in output chunks (e.g., for validation).
        tokenizer (AutoTokenizer): Tokenizer to encode text.
        max_length (int): Maximum tokens per chunk (including special tokens).
        overlap (int): Number of tokens to overlap between chunks.
        device (str): Device for tensors ("cpu" or "cuda").

    Returns:
        List[Dict]: List of tokenized chunks with input_ids, attention_mask, text, etc.

    Raises:
        ValueError: If max_length is too small, overlap is invalid, or text is empty.
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty")
    if max_length < 3:
        raise ValueError("max_length must be at least 3 to accommodate special tokens")
    if overlap < 0 or overlap >= max_length - 2:
        raise ValueError("overlap must be non-negative and less than max_length - 2")

    # Tokenize the entire text without truncation
    encoded = tokenizer(
        text, add_special_tokens=False, return_tensors="pt", return_attention_mask=True
    )

    input_ids = encoded["input_ids"][0].to(device)
    attention_mask = encoded["attention_mask"][0].to(device)

    # Account for [CLS] and [SEP] in max_length
    chunk_size = max_length - 2
    chunks = []
    i = 0

    while i < len(input_ids):
        end = min(i + chunk_size, len(input_ids))

        # Skip very short final chunks (optional)
        if end == len(input_ids) and end - i < 10:  # Arbitrary threshold
            break

        # Extract chunk
        chunk_ids = input_ids[i:end]
        chunk_mask = attention_mask[i:end]

        # Decode chunk text (exclude special tokens later)
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        # Add special tokens (handle missing cls/sep gracefully)
        cls_id = (
            tokenizer.cls_token_id or tokenizer.bos_token_id or tokenizer.pad_token_id
        )
        sep_id = (
            tokenizer.sep_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id
        )
        if cls_id is None or sep_id is None:
            raise ValueError(
                "Tokenizer lacks required special tokens (cls/sep or bos/eos)"
            )

        chunk_ids = torch.cat(
            [
                torch.tensor([cls_id], dtype=chunk_ids.dtype, device=device),
                chunk_ids,
                torch.tensor([sep_id], dtype=chunk_ids.dtype, device=device),
            ]
        )
        chunk_mask = torch.cat(
            [
                torch.tensor([1], dtype=chunk_mask.dtype, device=device),
                chunk_mask,
                torch.tensor([1], dtype=chunk_mask.dtype, device=device),
            ]
        )

        # Optional padding to max_length
        padding_length = max_length - len(chunk_ids)
        if padding_length > 0:
            chunk_ids = torch.cat(
                [
                    chunk_ids,
                    torch.zeros(padding_length, dtype=chunk_ids.dtype, device=device),
                ]
            )
            chunk_mask = torch.cat(
                [
                    chunk_mask,
                    torch.zeros(padding_length, dtype=chunk_mask.dtype, device=device),
                ]
            )

        # Prepare chunk dictionary
        chunk_dict = {
            "input_ids": chunk_ids,
            "attention_mask": chunk_mask,
            "text": chunk_text,
            "is_valid": is_valid,
        }

        # Add token_type_ids only if required
        if "token_type_ids" in tokenizer.model_input_names:
            chunk_dict["token_type_ids"] = torch.zeros_like(chunk_ids)

        chunks.append(chunk_dict)

        # Move to next chunk with overlap
        i += chunk_size - overlap

    if not chunks:
        raise ValueError("No valid chunks produced; input may be too short")

    return chunks


def chunk_text(
    text: str,
    is_valid: bool,
    labels: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    overlap: int = 50,
    candidate_label_ids: list = None,  # fixed-size candidate-budget label ids, candidate mode only
    candidate_mask: list = None,  # fixed-size candidate-budget real/padding mask, candidate mode only
):
    """
    Split text into chunks that fit within max_length tokens, with optional overlap.

    Args:
        text (str): Input text to chunk.
        is_valid (bool): Whether the text is in training or validation set
        labels (str): The multiple labels seperated by semicolon
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
        return_attention_mask=True,
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

        # Decode chunk text (exclude special tokens later)
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        # Add special tokens
        cls_id = tokenizer.cls_token_id or tokenizer.bos_token_id
        sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id
        chunk_ids = torch.cat(
            [
                torch.tensor([cls_id], dtype=chunk_ids.dtype),
                chunk_ids,
                torch.tensor([sep_id], dtype=chunk_ids.dtype),
            ]
        )
        chunk_mask = torch.cat(
            [
                torch.tensor([1], dtype=chunk_mask.dtype),
                chunk_mask,
                torch.tensor([1], dtype=chunk_mask.dtype),
            ]
        )

        # Pad to max_length if needed
        padding_length = max_length - len(chunk_ids)
        if padding_length > 0:
            chunk_ids = torch.cat(
                [chunk_ids, torch.zeros(padding_length, dtype=chunk_ids.dtype)]
            )
            chunk_mask = torch.cat(
                [chunk_mask, torch.zeros(padding_length, dtype=chunk_mask.dtype)]
            )

        # Store chunk
        chunk_dict = {
            "input_ids": chunk_ids,
            "attention_mask": chunk_mask,
            "text": chunk_text,
            "token_type_ids": (
                torch.zeros_like(chunk_ids)
                if "token_type_ids" in tokenizer.model_input_names
                else None
            ),
            "is_valid": is_valid,
            "labels": labels,
        }
        # Only add these keys in candidate mode. If every chunk dict always
        # carried them (even as None in label mode), Dataset.from_list would
        # infer a schema that includes candidate_label_ids/candidate_mask
        # columns regardless of label_space -- exactly the bug found and
        # fixed in multilabel_classify.py's chunk_text (see
        # README_AMAZON3M.md, Phase 7's KeyError writeup).
        if candidate_label_ids is not None:
            chunk_dict["candidate_label_ids"] = candidate_label_ids
            chunk_dict["candidate_mask"] = candidate_mask
        chunks.append(chunk_dict)

        # Move to next chunk with overlap
        i += chunk_size - overlap

    return chunks


def load_cluster_assignments(base_dir):
    "Load the flat label clustering built by scripts/build_label_clusters.py."
    with open(os.path.join(base_dir, "cluster_assignments.json"), "r") as f:
        cluster_assignments = json.load(f)
    return cluster_assignments["cluster_id"], cluster_assignments["num_clusters"]


def build_candidate_arrays(label_vectors, mlb_classes, base_dir, budget=256, seed=42):
    """Build a fixed-size, per-document candidate set for the full-scale
    candidate-generator plan (see README_AMAZON3M.md): true positive label
    indices plus same-cluster hard negatives (labels sharing a cluster with
    at least one of the doc's true positives), padded/truncated to `budget`.
    This is what lets Stage 4's "fine" LTRModel training score a bounded set
    of labels per document instead of every one of Amazon-3M's ~2.81M
    labels. Local duplicate of multilabel_classify.py's function of the same
    name (no cross-script imports between these driver scripts, matching
    this repo's existing convention, e.g. load_cluster_assignments above).

    `label_vectors` (sparse, shape (num_docs, num_labels)) and `mlb_classes`
    must use the SAME label ordering as the plain (non-cluster) Stage 3
    bootstrap's df_lbs (lbl -> lbl_val) and the clustering artifacts under
    `base_dir` -- see preprocess_data's candidate-mode branch, which builds
    `mlb_classes` from `df_lbs` for exactly this reason.

    Returns (candidate_label_ids, candidate_targets, candidate_mask), each a
    numpy array of shape (num_docs, budget). LTRModel doesn't need
    candidate_targets (its ground truth comes from lookup_table, not a
    doc-level target vector) -- kept for parity with the classifier-head
    version and to keep the two copies easy to diff against each other.
    """
    cluster_ids, num_clusters = load_cluster_assignments(base_dir)
    assert len(cluster_ids) == len(mlb_classes), (
        f"cluster_assignments.json has {len(cluster_ids)} entries, expected "
        f"{len(mlb_classes)} -- was build_label_clusters.py run against a "
        f"stale labels_partition.json, or this Stage 3 bootstrap run in "
        f"--label_space=cluster mode instead of plain 'label' mode?"
    )
    cluster_to_labels = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        cluster_to_labels[cid].append(idx)

    rng = random.Random(seed)
    label_vectors = label_vectors.tocsr()
    num_docs = label_vectors.shape[0]

    candidate_label_ids = np.zeros((num_docs, budget), dtype=np.int64)
    candidate_targets = np.zeros((num_docs, budget), dtype=np.float32)
    candidate_mask = np.zeros((num_docs, budget), dtype=bool)

    for row in range(num_docs):
        pos_idx = label_vectors.indices[
            label_vectors.indptr[row]: label_vectors.indptr[row + 1]
        ].tolist()
        pos_idx = pos_idx[:budget]  # truncate in the rare case a doc has more true labels than the whole budget
        pos_set = set(pos_idx)

        touched_clusters = sorted(set(cluster_ids[i] for i in pos_idx))
        neg_pool = [
            i for c in touched_clusters for i in cluster_to_labels[c] if i not in pos_set
        ]
        rng.shuffle(neg_pool)
        n_neg = min(len(neg_pool), budget - len(pos_idx))
        negs = neg_pool[:n_neg]

        slots = pos_idx + negs
        n_slots = len(slots)
        candidate_label_ids[row, :n_slots] = slots
        candidate_mask[row, :n_slots] = True
        candidate_targets[row, : len(pos_idx)] = 1.0

    return candidate_label_ids, candidate_targets, candidate_mask


def preprocess_data(
    df: pd.DataFrame,
    scored_tokens: torch.Tensor,
    df_toks: pd.DataFrame,
    df_lbs: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_len: int = 512,
    batch_size: int = 4,
    save_to_disk: str = None,
    device: str = "cuda",
    label_space: str = "label",
    candidate_budget: int = 256,
    base_dir: str = None,
) -> Dataset:
    """
    Create a HuggingFace Dataset for Learning-to-Rank, with batched tokenization and relevance score computation.

    `label_space="candidate"` restricts each document's forward pass/loss to
    a bounded per-document candidate set (true positives + same-cluster hard
    negatives, fixed size `candidate_budget`) instead of scoring every label
    in `df_lbs` -- see `build_candidate_arrays`. Requires `base_dir` to
    contain `cluster_assignments.json`/`cluster_to_labels.json`
    (`scripts/build_label_clusters.py`), and `df_lbs` to come from a plain
    (non-cluster) Stage 3 bootstrap run, so label ids line up with the
    clustering (see README_AMAZON3M.md's full-scale candidate-generator
    plan).

    Args:
        df (pd.DataFrame): DataFrame with 'text' column containing raw text.
        scored_tokens (torch.Tensor): Tensor of shape (num_labels, num_tokens, 4) with [token, label, rank, relevance_score].
        df_toks (pd.DataFrame): DataFrame with columns ['token', 'tok_val'] mapping token IDs to token values.
        df_lbs (pd.DataFrame): DataFrame with columns ['lbl', 'lbl_val'] mapping label IDs to label values.
        tokenizer (AutoTokenizer): Tokenizer for the pretrained model.
        max_len (int): Maximum sequence length for tokenization. Default is 128.
        batch_size (int): Batch size for map processing. Default is 4.
        num_proc (int): Number of processes for map parallelization. Default is 4.
        save_to_disk (str, optional): Path to save the dataset to disk. Default is None.
        device (str): Device for tensor operations (e.g., 'cuda', 'cpu', 'cuda:0'). Default is 'cuda'.

    Returns:
        Dataset: HuggingFace Dataset with columns ['text', 'input_ids', 'attention_mask', 'relevance_scores'].
    """
    # Validate inputs
    if "text" not in df.columns:
        raise ValueError("df must contain 'text' column")
    if scored_tokens.shape[2] != 4:
        raise ValueError(
            f"Expected scored_tokens shape (num_labels, num_tokens, 4), got {scored_tokens.shape}"
        )
    if not all(col in df_toks.columns for col in ["token", "tok_val"]):
        raise ValueError("df_toks must contain columns: ['token', 'tok_val']")
    if not all(col in df_lbs.columns for col in ["lbl", "lbl_val"]):
        raise ValueError("df_lbs must contain columns: ['lbl', 'lbl_val']")
    if not torch.isfinite(scored_tokens[:, :, 3]).all():
        raise ValueError("scored_tokens contains invalid (inf/nan) relevance scores")

    num_labels = scored_tokens.shape[0]
    device = torch.device(device)
    scored_tokens = scored_tokens.to(device)

    # Verify uniqueness of token IDs per label
    logging.info("🔍 Verifying uniqueness of token IDs per label in scored_tokens...")
    for label_idx in tqdm(
        range(num_labels), desc="Checking token uniqueness per label"
    ):
        token_ids = scored_tokens[label_idx, :, 0]
        unique_tokens = torch.unique(token_ids)
        if len(unique_tokens) < len(token_ids):
            raise ValueError(
                f"scored_tokens[{label_idx}, :, 0] contains duplicate token IDs "
                f"({len(token_ids) - len(unique_tokens)} duplicates found). "
                "Each label must have unique token IDs for correct relevance score aggregation."
            )
    logging.info("✅ All labels have unique token IDs in scored_tokens.")

    # Build Lookup Table
    logging.info(
        "🚀 Building a 2D lookup table for efficient token-to-relevance mapping across all labels! 🚀"
    )
    logging.info(f"🔢 Total labels = {num_labels}")
    logging.info(
        f"🔍 Precomputing token indices and corresponding relevance_values for each label..."
    )
    tokens = scored_tokens[:, :, 0].long()  # Shape: (num_labels, num_tokens)
    vocab_size = tokens.max().item() + 1
    relevance_values = scored_tokens[:, :, 3]  # Shape: (num_labels, num_tokens)
    lookup_table = torch.zeros((vocab_size, num_labels), device=device)
    logging.info(
        f"📊 Lookup table dimensions: {vocab_size} vocabulary size × {num_labels} labels"
    )
    logging.info(
        "⚡ This approach eliminates token comparison broadcasting and provides O(1) lookup time for relevance scores!"
    )
    logging.info("🧮 Processing relevance calculations vectorized for maximum speed 🔥")
    """
    🔍 Efficient relevance computation using 2D lookup table
    Instead of comparing each input token against all label tokens (which requires expensive
    broadcasting operations), we build a lookup table where:
    - Each row corresponds to a token ID (0 to vocab_size-1)
    - Each column corresponds to a label (0 to num_labels-1)
    - Each cell contains the relevance of that token for that label

    This approach has several advantages:
    1. O(1) lookup time for relevance scores (vs. O(n) for comparison-based approaches)
    2. Eliminates the need for expensive broadcasting and comparison operations
    3. Processes all labels and tokens with a single vectorized operation
    4. Reduces both memory usage and computation time for large datasets

    The approach works in two phases:
    1. Build the lookup table by populating relevance values for all (token, label) pairs
    2. Use the lookup table to directly map from input_ids to relevance scores

    When using the lookup approach, we ensure all input tokens are within the valid
    vocabulary range to prevent out-of-bounds errors during indexing.
    """

    # Fill the lookup table for all labels
    for label_idx in tqdm(range(num_labels), desc="Filling the lookup table per label"):
        label_tokens = tokens[label_idx, :].long()  # Shape: (num_tokens)
        label_relevance = relevance_values[label_idx, :]  # Shape: (num_tokens)
        lookup_table[:, label_idx].index_put_((label_tokens,), label_relevance)

    # Verify token mappings
    def verify_token_mappings(
        df_toks: pd.DataFrame, tokenizer: AutoTokenizer, sample_texts: List[str]
    ) -> None:
        logging.info(f"🔍 Verifying token mappings with {len(sample_texts)} samples...")
        df_toks_mapping = dict(zip(df_toks["token"], df_toks["tok_val"]))
        mismatches = []

        for text in tqdm(
            sample_texts, desc="Verifying tokenizer and df_toks compatibility"
        ):
            encoding = tokenizer(text, return_tensors="pt")
            input_ids = encoding["input_ids"].squeeze(0)
            token_sample_indices = random.sample(
                range(len(input_ids)), min(5000, len(input_ids))
            )

            for token_idx in token_sample_indices:
                try:
                    # Check if token_idx is within bounds
                    if token_idx >= len(input_ids):
                        print(
                            f"Warning: token_idx {token_idx} out of bounds for input_ids of length {len(input_ids)}"
                        )
                        continue

                    # Safely extract token_id
                    try:
                        token_id = int(input_ids[token_idx].item())
                    except (AttributeError, ValueError) as e:
                        print(
                            f"Error extracting token_id at index {token_idx}: {str(e)}"
                        )
                        continue

                    # Safely get value from mapping dictionary
                    df_toks_value = df_toks_mapping.get(token_id, None)

                    # Safely get value from tokenizer
                    # NOTE: len(tokenizer), not tokenizer.vocab_size -- vocab_size
                    # reports only the base vocab and excludes added/reserved
                    # special tokens (e.g. Llama-3.1: vocab_size=128000 vs
                    # len(tokenizer)=128256), while df_toks (built in Stage 3)
                    # spans the full len(tokenizer) id space. Using vocab_size
                    # here manufactured false mismatches for every added-token id.
                    tokenizer_value = None
                    if 0 <= token_id < len(tokenizer):
                        try:
                            tokenizer_value = tokenizer.convert_ids_to_tokens(token_id)
                        except Exception as e:
                            print(
                                f"Error converting token_id {token_id} to token: {str(e)}"
                            )
                    else:
                        print(
                            f"Token ID {token_id} out of range (valid range: 0-{len(tokenizer)-1})"
                        )

                    # More robust comparison
                    if df_toks_value is None and tokenizer_value is None:
                        # Both are None, they're effectively equal
                        continue
                    elif df_toks_value is None or tokenizer_value is None:
                        # One is None but the other isn't, they're different
                        mismatches.append((token_id, df_toks_value, tokenizer_value))
                    else:
                        # Compare strings after normalizing
                        if str(df_toks_value).strip() != str(tokenizer_value).strip():
                            mismatches.append(
                                (token_id, df_toks_value, tokenizer_value)
                            )

                except Exception as e:
                    # Catch-all for any unexpected errors
                    print(
                        f"Unexpected error processing token_idx {token_idx}: {str(e)}"
                    )
                    continue

        # # Check top 5 frequent tokens
        # top_n_tokens = self.df_toks['token'].value_counts().head(5).index
        # for token_id in top_n_tokens:
        #     if token_id in df_toks_mapping and token_id in self.tokenizer.vocab:
        #         df_toks_value = df_toks_mapping[token_id]
        #         tokenizer_value = self.tokenizer.convert_ids_to_tokens(token_id)
        #         if df_toks_value != tokenizer_value:
        #             mismatches.append((token_id, df_toks_value, tokenizer_value))

        if mismatches:
            logging.warning(f"⚠️ Found {len(mismatches)} token mapping inconsistencies:")
            for token_id, df_val, tok_val in mismatches[:10]:
                logging.warning(
                    f"🔴 Token ID {token_id}: df_toks='{df_val}', tokenizer='{tok_val}'"
                )
            if len(tokenizer) != len(df_toks):
                logging.error(
                    f"❌ Tokenizer's vocab size = {len(tokenizer)}, while df_tok's size = {len(df_toks)}"
                )
            raise ValueError(
                "❌ Token mapping inconsistencies found between df_toks and tokenizer."
            )
        logging.info("✅ Token mappings verification completed successfully! 🎉")

    sample_texts = df["text"].sample(n=min(10, len(df)), random_state=42).tolist()
    verify_token_mappings(df_toks, tokenizer, sample_texts)

    # Check if dataset exists at save_to_disk
    # save_file = Path(save_to_disk)/"l2r_dataset"
    if save_to_disk and os.path.exists(save_to_disk):
        logging.info(f"📂 Found existing dataset at {save_to_disk}. Loading...")
        dataset = Dataset.load_from_disk(save_to_disk)
        logging.info(f"✅ Loaded dataset with {len(dataset)} samples")
        # Verify expected columns
        expected_columns = {"input_ids", "attention_mask"}
        if not all(col in dataset.column_names for col in expected_columns):
            raise ValueError(
                f"Loaded dataset missing required columns. Expected {expected_columns}, got {dataset.column_names}"
            )
        dataset.set_format("torch", columns=["input_ids", "attention_mask"])
        return dataset, num_labels, lookup_table

    if label_space == "candidate":
        # mlb_classes must line up with df_lbs's own 'lbl' ids (which is what
        # lookup_table's columns and every candidate_label_ids value actually
        # index into) -- df_lbs comes from a plain (non-cluster) Stage 3
        # bootstrap run, whose 'lbl' ids are assigned in the exact order of
        # labels_partition.json's "classes" list (see bootl2r.py), the same
        # ordering cluster_assignments.json was built against. Sorting by
        # 'lbl' here (rather than trusting file row order) makes that
        # assumption explicit rather than implicit.
        mlb_classes = df_lbs.sort_values("lbl")["lbl_val"].astype(str).tolist()
        raw_labels = df["labels"].apply(lambda x: x.split(";") if x else []).tolist()
        mlb = MultiLabelBinarizer(classes=mlb_classes, sparse_output=True)
        label_vectors = mlb.fit_transform(raw_labels)
        candidate_label_ids, _candidate_targets, candidate_mask = build_candidate_arrays(
            label_vectors, mlb_classes, base_dir, budget=candidate_budget
        )
        logging.info(
            f"[candidate] Built {candidate_label_ids.shape[0]} per-doc candidate sets "
            f"(budget={candidate_budget}); avg real (positive+negative) slots per doc: "
            f"{candidate_mask.sum(axis=1).mean():.1f}"
        )
        df = df.copy()
        df["candidate_label_ids"] = list(candidate_label_ids)
        df["candidate_mask"] = list(candidate_mask)

    # Create HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Define batched processing function
    def process_batch_old(examples: Dict[str, List]) -> Dict[str, List]:
        texts = examples["text"]
        encodings = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)  # Shape: (batch_size, max_len)
        attention_mask = encodings["attention_mask"].to(
            device
        )  # Shape: (batch_size, max_len)

        # Compute relevance scores for the batch
        # relevance_scores = lookup_table[input_ids, :]

        # Convert tensors to lists for Dataset compatibility
        return {
            "input_ids": input_ids.cpu().numpy().tolist(),
            "attention_mask": attention_mask.cpu().numpy().tolist(),
        }

    def process_batch(examples):
        all_chunks = []
        candidate_cols = (
            zip(examples["candidate_label_ids"], examples["candidate_mask"])
            if label_space == "candidate"
            else itertools.repeat((None, None))
        )
        for (text, is_valid, labels), (cand_ids, cand_mask) in zip(
            zip(examples["text"], examples["is_valid"], examples["labels"]), candidate_cols
        ):
            chunks = chunk_text(
                text, is_valid, labels, tokenizer, max_len, overlap=10,
                candidate_label_ids=cand_ids, candidate_mask=cand_mask,
            )

            # check that every single chunk has the same `is_valid` and 'labels' field
            is_valids = [chunk["is_valid"] for chunk in chunks]
            assert all(x == is_valids[0] for x in is_valids)
            the_labels = [chunk["labels"] for chunk in chunks]
            assert all(x == the_labels[0] for x in the_labels)

            all_chunks.append(chunks)
        return {"chunks": all_chunks}

    # Apply batched processing
    logging.info("🔄 Processing dataset with map...")
    start_time = time.time()

    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        # remove_columns=['text'],  # Remove original text to save memory
        desc="Processing texts and saving relevance scores",
    )

    # Flatten the chunks into a list of dictionaries
    flattened_chunks = [
        chunk
        for data in tqdm(dataset, desc="Flattening chunks")
        for chunk in data["chunks"]
    ]
    dataset = Dataset.from_list(flattened_chunks)

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"✅ Dataset built in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # Set format for PyTorch compatibility
    # dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    format_columns = [
        "input_ids",
        "attention_mask",
        "is_valid",
        "text",
        "token_type_ids",
        "labels",
    ]
    if label_space == "candidate":
        format_columns += ["candidate_label_ids", "candidate_mask"]
    dataset.set_format("torch", columns=format_columns)

    # Split based on is_valid column
    train_dataset = dataset.filter(
        lambda example: not example["is_valid"], desc="Filtering train set"
    )
    test_dataset = dataset.filter(
        lambda example: example["is_valid"], desc="Filtering test set"
    )

    # Remove the 'is_valid' column from the final datasets (optional)
    train_dataset = train_dataset.remove_columns("is_valid")
    test_dataset = test_dataset.remove_columns("is_valid")

    # Create a DatasetDict for train and test
    split_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    logging.info(f"The size of Training set: {len(train_dataset)} 🏋️ (Training Data)")
    logging.info(f"The size of Evaluation set: {len(test_dataset)} 🔍 (Test Data)")

    # Save to disk if specified
    if save_to_disk:
        logging.info(f"💾 Saving dataset to {save_to_disk}...")
        dataset.save_to_disk(save_to_disk)
        logging.info("✅ Dataset saved successfully!")

    logging.info(
        f"🚀 Created HuggingFace Dataset with {len(dataset)} samples, {num_labels} labels"
    )

    return split_dataset, num_labels, lookup_table


def lambdarank_loss_old(
    pred_scores: torch.Tensor, true_scores: torch.Tensor
) -> torch.Tensor:
    """
    Computes LambdaRank loss, weighting pairwise gradients by the change in NDCG.

    Args:
        pred_scores (torch.Tensor): Predicted relevance scores, shape (batch_size, max_len, num_labels).
        true_scores (torch.Tensor): Ground-truth relevance scores, shape (batch_size, max_len, num_labels).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    batch_size, max_len, num_labels = pred_scores.shape
    loss = 0.0
    ranks = torch.arange(max_len, device=pred_scores.device, dtype=torch.float) + 1
    dcg_weights = 1.0 / torch.log2(ranks + 1)
    dcg_diff = dcg_weights.unsqueeze(1) - dcg_weights.unsqueeze(0)

    for l in range(num_labels):
        pred = pred_scores[:, :, l]  # Shape: (batch_size, max_len)
        true = true_scores[:, :, l]  # Shape: (batch_size, max_len)
        pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
        true_diff = true.unsqueeze(2) - true.unsqueeze(1)
        pair_mask = (true_diff > 0).float()
        ndcg_delta = torch.abs(dcg_diff) * pair_mask
        sigmoid_input = -pred_diff
        lambda_grad = torch.sigmoid(sigmoid_input) * ndcg_delta
        loss += lambda_grad.sum()

    return (
        loss / (batch_size * num_labels)
        if loss > 0
        else torch.tensor(0.0, device=pred_scores.device)
    )


def lambdarank_loss(
    pred_scores: torch.Tensor, true_scores: torch.Tensor
) -> torch.Tensor:
    """
    Computes LambdaRank loss, weighting pairwise gradients by the change in NDCG, fully vectorized.

    Args:
        pred_scores (torch.Tensor): Predicted relevance scores, shape (batch_size, max_len, num_labels).
        true_scores (torch.Tensor): Ground-truth relevance scores, shape (batch_size, max_len, num_labels).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    batch_size, max_len, num_labels = pred_scores.shape

    # Reshape to combine batch and label dimensions
    pred = pred_scores.permute(0, 2, 1).reshape(
        batch_size * num_labels, max_len
    )  # Shape: (batch_size * num_labels, max_len)
    true = true_scores.permute(0, 2, 1).reshape(
        batch_size * num_labels, max_len
    )  # Shape: (batch_size * num_labels, max_len)

    # Compute pairwise differences
    pred_diff = pred.unsqueeze(2) - pred.unsqueeze(
        1
    )  # Shape: (batch_size * num_labels, max_len, max_len)
    true_diff = true.unsqueeze(2) - true.unsqueeze(
        1
    )  # Shape: (batch_size * num_labels, max_len, max_len)

    # Create pair mask for relevant pairs (where true relevance differs)
    pair_mask = (
        true_diff > 0
    ).float()  # Shape: (batch_size * num_labels, max_len, max_len)

    # Compute NDCG delta
    ranks = torch.arange(max_len, device=pred_scores.device, dtype=torch.float) + 1
    dcg_weights = 1.0 / torch.log2(ranks + 1)  # Shape: (max_len,)
    dcg_diff = dcg_weights.unsqueeze(1) - dcg_weights.unsqueeze(
        0
    )  # Shape: (max_len, max_len)
    ndcg_delta = (
        torch.abs(dcg_diff) * pair_mask
    )  # Shape: (batch_size * num_labels, max_len, max_len)

    # Compute LambdaRank gradients
    sigmoid_input = -pred_diff  # Negative for pairs where i > j
    lambda_grad = (
        torch.sigmoid(sigmoid_input) * ndcg_delta
    )  # Shape: (batch_size * num_labels, max_len, max_len)

    # Sum gradients and normalize
    loss = lambda_grad.sum()
    loss = (
        loss / (batch_size * num_labels)
        if loss > 0
        else torch.tensor(0.0, device=pred_scores.device)
    )

    return loss


def compute_trainable_params(model):
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / total_params

    logging.info(
        f"📊 trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percentage:.4f}"
    )
    return trainable_params, total_params


# Function to define the custom model
def define_model(base_model, num_labels, lookup_table, label_embed_rank=None):
    """Define the Learning-to-Rank model based on a pretrained transformer, outputting per-token relevance scores per label.

    `label_embed_rank`, when set, factorizes LTRModel's per-label embedding
    table into a low-rank base plus one shared up-projection instead of a
    full-rank (num_labels, hidden_size) table -- see LTRModel.__init__ and
    README_AMAZON3M.md's full-scale candidate-generator plan.
    """

    # Create config instance with all base_model.config attributes plus add your custom attributes
    config_dict = base_model.config.to_dict()
    ground_model_name_or_path=config_dict['_name_or_path']
    config = LTRConfig(num_labels=num_labels, ground_model_name_or_path=ground_model_name_or_path,
                        label_embed_rank=label_embed_rank, **config_dict)
    # Modify the config's __dict__ to include your custom attributes
    # This ensures they'll be saved with the model
    config_dict = config.__dict__
    config_dict["num_labels"] = num_labels
    config_dict["ground_model_name_or_path"] = ground_model_name_or_path
    config_dict["label_embed_rank"] = label_embed_rank

    # Initialize LTRModel with config
    model = LTRModel(config)
    model.ground_model = base_model  # Set the base model manually
    model.lookup_table = lookup_table  # Set the lookup table

    # Register the model
    logging.info("🔧 Registering LTRModel with transformers AutoModel 🚀")
    AutoModel.register(LTRConfig, LTRModel, exist_ok=True)

    return model


def log_training_args(training_args, **kwargs):
    """
    Log training arguments with informative emojis in a tabular format

    Args:
        training_args (TrainingArguments): Training arguments
        **kwargs: Optional keyword arguments including train_dataset, eval_dataset
    """

    # Extract datasets from kwargs
    train_dataset = kwargs.get("train_dataset", None)
    eval_dataset = kwargs.get("eval_dataset", None)

    # Emoji Dictionary for Training Arguments
    emoji_map = {
        "output_dir": "📁",  # Folder for output
        "num_train_epochs": "🔁",  # Repeat/Cycle
        "batch_size_train": "🏋️",  # Training (weight lifter)
        "batch_size_eval": "🔍",  # Evaluation (magnifying glass)
        "learning_rate": "🚀",  # Rocket (speed/acceleration)
        "warmup_steps": "🌅",  # Sunrise (warming up)
        "gradient_accumulation": "📊",  # Bar chart (accumulation)
        "save_strategy": "💾",  # Save disk
        "evaluation_strategy": "📊",  # Bar chart (metrics)
        "metric": "🎯",  # Target/Goal
        "logging": "📝",  # Writing/Logging
        "hub_push": "🌐",  # Globe (pushing to hub)
        "dataset": "📊",  # Dataset size
        "steps": "🔢",  # Steps calculation
    }

    # Calculate training steps
    if train_dataset is not None:
        # Calculate total training steps
        total_train_samples = len(train_dataset)
        train_batch_size = (
            training_args.per_device_train_batch_size * training_args.world_size
        )
        gradient_accumulation_steps = training_args.gradient_accumulation_steps

        # Calculate total steps per epoch and total training steps
        steps_per_epoch = math.floor(
            total_train_samples / (train_batch_size * gradient_accumulation_steps)
        )
        total_training_steps = steps_per_epoch * training_args.num_train_epochs
    else:
        steps_per_epoch = "N/A"
        total_training_steps = "N/A"

    # Calculate evaluation steps
    if eval_dataset is not None:
        # Calculate total evaluation steps
        total_eval_samples = len(eval_dataset)
        eval_batch_size = (
            training_args.per_device_eval_batch_size * training_args.world_size
        )

        # Calculate steps for evaluation
        evaluation_steps = math.floor(total_eval_samples / eval_batch_size)
    else:
        evaluation_steps = "N/A"

    # Prepare data for tabulation
    training_details = [
        [emoji_map["output_dir"], "Output Directory", training_args.output_dir],
        [
            emoji_map["num_train_epochs"],
            "Training Epochs",
            training_args.num_train_epochs,
        ],
        [
            emoji_map["batch_size_train"],
            "Train Batch Size",
            training_args.per_device_train_batch_size,
        ],
        [
            emoji_map["batch_size_eval"],
            "Eval Batch Size",
            training_args.per_device_eval_batch_size,
        ],
        [
            emoji_map["gradient_accumulation"],
            "Gradient Accumulation Steps",
            training_args.gradient_accumulation_steps,
        ],
        [emoji_map["learning_rate"], "Learning Rate", training_args.learning_rate],
        [emoji_map["warmup_steps"], "Warmup Steps", training_args.warmup_steps],
        [emoji_map["save_strategy"], "Save Strategy", training_args.save_strategy],
        [
            emoji_map["save_strategy"],
            "Save Total Limit",
            training_args.save_total_limit,
        ],
        [
            emoji_map["evaluation_strategy"],
            "Evaluation Strategy",
            training_args.evaluation_strategy,
        ],
        [emoji_map["metric"], "Best Model Metric", training_args.metric_for_best_model],
        [
            emoji_map["logging"],
            "Logging Strategy",
            f"{training_args.logging_strategy} (every {training_args.logging_steps} steps)",
        ],
        [emoji_map["hub_push"], "Push to Hub", training_args.push_to_hub],
        [
            emoji_map["hub_push"],
            "Hub Model ID",
            training_args.hub_model_id or "Not Set",
        ],
    ]

    # Add steps information
    training_details.extend(
        [
            [emoji_map["steps"], "Steps per Epoch", steps_per_epoch],
            [emoji_map["steps"], "Total Training Steps", total_training_steps],
            [emoji_map["steps"], "Evaluation Steps", evaluation_steps],
        ]
    )

    # Add dataset size information if datasets are provided
    if train_dataset is not None:
        training_details.append(
            [
                emoji_map["dataset"],
                "Training Dataset Size",
                f"{len(train_dataset)} samples 🏋️",
            ]
        )

    if eval_dataset is not None:
        training_details.append(
            [
                emoji_map["dataset"],
                "Evaluation Dataset Size",
                f"{len(eval_dataset)} samples 🔍",
            ]
        )

    # Create formatted table
    table = tabulate(
        training_details,
        headers=["🌟 Emoji", "🏷️ Parameter", "📊 Value"],
        tablefmt="pretty",
    )

    # Log the table
    logging.info("\n📋 Training Configuration 📋\n%s", table)


def log_training_configuration(training_args, **kwargs):
    """
    Comprehensive logging of training configuration
    """
    # Log hardware details
    logging.info(
        f"🖥️ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    logging.info(f"🔋 CUDA Available: {torch.cuda.is_available()}")
    logging.info(f"💾 CUDA Device Count: {torch.cuda.device_count()}")

    # Log training arguments table
    log_training_args(training_args, **kwargs)


# Function to Define Training Arguments
def define_training_args(
    output_dir="path/to/output_dir",
    task_hub_model_id="deb101/your-downstream_task",
    hub_token=None,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=500,
    metric_for_best_model: str = "ndcg@k",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    push_to_hub=True,
):
    """Define training arguments for the Trainer."""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True,  # matches the bf16 model loads + bnb compute dtype +
        # accelerate mixed_precision=bf16; fp16 here was inconsistent with all
        # three and, per Hugging Face, fp16 and bf16 are mutually exclusive
        # (setting fp16=True while accelerate is configured for bf16 either
        # errors or silently disables mixed precision).
        save_safetensors=False,  # Phi-3-small ties embed_tokens/lm_head
        # (tie_word_embeddings=True, unlike Mistral/Llama); safetensors
        # can't serialize tensors that share memory. torch.save
        # (pytorch_model.bin) handles aliased tensors natively. Harmless
        # no-op for untied Mistral/Llama. Also needed the ground_model
        # save inside LTRModel.save_pretrained (xcube/l2r/models/core.py)
        # fixed separately, since it wasn't forwarding this kwarg through.
        warmup_steps=warmup_steps,
        # Save and evaluate at the end of each epoch
        save_strategy="epoch",
        # save_strategy="no",
        save_total_limit=10,
        evaluation_strategy="epoch",
        # Learning rate scheduler settings
        # lr_scheduler_type="constant",
        # Save only the best model based on a metric
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,  # Monitor accuracy (higher is better)
        greater_is_better=True,  # For accuracy like metrics, higher values are better
        logging_strategy="steps",  # Log training loss every 10 steps (no eval),
        logging_steps=10,
        report_to="none",
        push_to_hub=push_to_hub,
        hub_model_id=task_hub_model_id,
        hub_strategy="end",
        hub_token=hub_token,
        disable_tqdm=True,
    )


def prepare_training_components(
    custom_model,
    dataset,
    output_dir,
    task_hub_model_id,
    hf_token,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=500,
    metric_for_best_model="ndcg",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    push_to_hub=True,
):
    """
    Prepare model and datasets for training with Accelerator.

    Args:
        custom_model: The model to be prepared for training
        dataset: Dictionary containing 'train' and 'test' datasets
        output_dir: Directory path for training outputs
        task_hub_model_id: Model ID for the task hub
        hf_token: Hugging Face authentication token

    Returns:
        tuple: (prepared_model, prepared_train_dataset, prepared_eval_dataset, training_args, accelerator)
    """
    # Define training arguments
    training_args = define_training_args(
        output_dir,
        task_hub_model_id,
        hf_token,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        metric_for_best_model=metric_for_best_model,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        push_to_hub=push_to_hub,
    )

    # Initialize Accelerator
    accelerator = Accelerator()

    # Prepare model and datasets
    prepared_model = accelerator.prepare(custom_model)
    prepared_train_dataset = accelerator.prepare(dataset["train"])
    prepared_eval_dataset = accelerator.prepare(dataset["test"])

    return (
        prepared_model,
        prepared_train_dataset,
        prepared_eval_dataset,
        training_args,
        accelerator,
    )


def generate_tokenrank_report(report_data: list, output_dir: str) -> None:
    """
    Generates and saves Markdown and JSON reports for token ranking results with timestamped filenames.

    Args:
        report_data: List of dictionaries containing token ranking data for each data point.
        output_dir: Directory to save the report files.
    """
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate Markdown report
    logging.info(f"🚀 Generating Token Ranking Report")

    report_md = "# 🏆 Token Ranking Report\n\n"

    # Use tqdm for progress tracking
    for idx, data_point in tqdm(
        enumerate(report_data),
        total=len(report_data),
        desc="📝 Processing Data Points",
        unit="point",
    ):
        report_md += f"## 📊 Data Point {idx + 1}: {data_point['text']}\n\n"
        report_md += (
            f"⭐ **Ground Truth Labels:** {data_point['ground_truth_labels']}\n\n"
        )

        for label_data in data_point["labels"]:
            report_md += f"### 🏷️ Label: {label_data['label_name']} (ID: {label_data['label_id']})\n"

            # Prepare table data
            table_data = []
            for rank, token in enumerate(label_data["top_tokens"], 1):
                table_data.append(
                    [
                        rank,
                        token["token_name"],
                        token["token_id"],
                        f"{token['score']:.4f}",
                    ]
                )

            # Use tabulate to create a pretty table
            table = tabulate(
                table_data,
                headers=["🏅 Rank", "🔤 Token Name", "🆔 Token ID", "📈 Score"],
                tablefmt="pipe",  # Markdown-compatible format
            )

            report_md += table + "\n\n"

    # Save Markdown report
    report_path = os.path.join(output_dir, f"tokenrank_report_{timestamp}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logging.info(f"📄 Token ranking report saved to {report_path} ✅")

    # Save JSON report for programmatic access
    json_report_path = os.path.join(output_dir, f"tokenrank_report_{timestamp}.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logging.info(f"🗂️ Token ranking JSON report saved to {json_report_path} ✅")


def visual_tokenrank_per_label(
    eval_pred: Tuple[torch.Tensor, torch.Tensor],
    batch: Dict[str, Any],
    df_toks: pd.DataFrame,
    df_lbs: pd.DataFrame,
    top_k: int = 50,
) -> Dict[str, Any]:
    """
    Ranks the top k tokens per label for each data point in the batch and maps to token/label names.

    Args:
        eval_pred: Tuple of (predictions, labels), where predictions and labels are tensors
                   of shape (batch_size, max_len, num_labels).
        batch: Dictionary containing 'input_ids', 'text', 'attention_mask', 'token_type_ids'.
        df_toks: DataFrame with columns ['token', 'tok_val'] mapping token IDs to token values.
        df_lbs: DataFrame with columns ['lbl', 'lbl_val'] mapping label IDs to label values.
        top_k: Number of top tokens to rank per label (default: 50).

    Returns:
        Dict[str, Any]: Dictionary containing top k tokens, scores, and names for each data point and label.
    """
    predictions, _ = eval_pred
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)

    batch_size, max_len, num_labels = predictions.shape
    report_data = []

    # Create mappings for efficiency
    token_map = {row["token"]: row["tok_val"] for _, row in df_toks.iterrows()}
    label_map = {row["lbl"]: row["lbl_val"] for _, row in df_lbs.iterrows()}

    # Iterate over the batch
    input_ids = batch["input_ids"]  # Shape: (batch_size, max_len)
    texts = batch["text"]  # List of strings
    labels = batch["labels"]

    for i in range(batch_size):
        data_point = {"text": texts[i], "ground_truth_labels": labels[i], "labels": []}
        pred_scores = predictions[i]  # Shape: (max_len, num_labels)
        input_ids_i = input_ids[i]  # Shape: (max_len,)

        for lbl_idx in tqdm(
            range(num_labels), leave=False, desc="Ranking input tokens per label"
        ):
            label_name = label_map.get(lbl_idx, f"Label_{lbl_idx}")
            scores = pred_scores[:, lbl_idx]  # Shape: (max_len,)
            top_k_indices = torch.topk(
                scores, k=min(top_k, max_len), dim=0
            ).indices  # Shape: (top_k,)
            top_k_scores = scores[top_k_indices].tolist()
            top_k_token_ids = input_ids_i[top_k_indices].tolist()

            # Map token IDs to token names
            top_k_tokens = [
                {
                    "token_id": token_id,
                    "token_name": token_map.get(token_id, f"Token_{token_id}"),
                    "score": score,
                }
                for token_id, score in zip(top_k_token_ids, top_k_scores)
            ]

            data_point["labels"].append(
                {
                    "label_id": lbl_idx,
                    "label_name": label_name,
                    "top_tokens": top_k_tokens,
                }
            )

        report_data.append(data_point)

    return {"report_data": report_data}


# =============================================================================
# Custom Classes
# =============================================================================


# Custom callback for logging
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_local_process_zero:
            # Only log as "Training metrics" if 'eval_' keys are absent
            if not any(k.startswith("eval_") for k in logs.keys()):
                log_message = "Training metrics: " + ", ".join(
                    f"{k}: {v}" for k, v in logs.items()
                )
                logging.info(log_message)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if (
            metrics is not None and state.is_local_process_zero
        ):  # Only log on main process
            log_message = "Evaluation metrics:\n" + "\n".join(
                f"{k}: {v}" for k, v in metrics.items()
            )
            logging.info(log_message)


class PrettyLoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_emoji = "🚂"  # Training locomotive
        self.eval_emoji = "🔍"  # Magnifying glass for evaluation

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_local_process_zero:
            # Only log as "Training metrics" if 'eval_' keys are absent
            if not any(k.startswith("eval_") for k in logs.keys()):
                # Format training metrics as a colored table
                table_data = [
                    [k, f"{v:.6f}" if isinstance(v, float) else v]
                    for k, v in logs.items()
                ]
                table = tabulate(
                    table_data,
                    headers=["Metric", "Value"],
                    tablefmt="grid",
                    colalign=("left", "right"),  # Left align keys, right align values
                )

                from colorama import Style

                log_message = f"\n{Fore.CYAN}{self.training_emoji} Training Metrics (Step {state.global_step}) {self.training_emoji}\n{table}{Style.RESET_ALL}"
                logging.info(log_message)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and state.is_local_process_zero:
            # Format evaluation metrics as a colored table
            table_data = [
                [k, f"{v:.6f}" if isinstance(v, float) else v]
                for k, v in metrics.items()
            ]
            table = tabulate(
                table_data,
                headers=["Metric", "Value"],
                tablefmt="grid",
                colalign=("left", "right"),  # Left align keys, right align values
            )

            from colorama import Style

            log_message = f"\n{Fore.YELLOW}{self.eval_emoji} Evaluation Metrics {self.eval_emoji}\n{table}{Style.RESET_ALL}"
            logging.info(log_message)


class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.training_bar = None
        # self.evaluation_bar = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        logging.info("Starting Training...")
        self.training_bar = tqdm(total=state.max_steps, desc="Training")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.training_bar is not None:
            self.training_bar.update(1)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.training_bar is not None:
            self.training_bar.close()
            logging.info("Training Completed!")


class ColoredProgressCallback(TrainerCallback):
    def __init__(self, train_color="green", eval_color="cyan"):
        """
        Initialize progress callback with customizable colors.

        Args:
            train_color: Color for training progress bar (default: "green")
            eval_color: Color for evaluation progress bar (default: "cyan")
        """
        self.training_bar = None
        self.evaluation_bar = None
        self.train_color = train_color
        self.eval_color = eval_color
        self.last_evaluated_step = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize training progress bar at the beginning of training."""
        logging.info("🚀 Starting Training...")
        self.training_bar = tqdm(
            total=state.max_steps,
            desc="🔄 Training",
            bar_format="{l_bar}%s{bar}%s{r_bar}"
            % ("\033[" + self._get_color_code(self.train_color), "\033[0m"),
            dynamic_ncols=True,
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Update training progress bar at the end of each step."""
        if self.training_bar is not None:
            self.training_bar.update(1)
            # Update description with loss if available
            if (
                hasattr(state, "log_history")
                and state.log_history
                and "loss" in state.log_history[-1]
            ):
                loss = state.log_history[-1]["loss"]
                self.training_bar.set_description(f"🔄 Training (Loss: {loss:.2e})")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Create or update evaluation progress bar during evaluation."""
        # Only create a new eval bar if we haven't created one since the last train step
        if self.last_evaluated_step != state.global_step:
            if self.evaluation_bar is not None:
                self.evaluation_bar.close()

            self.evaluation_bar = tqdm(
                total=100,
                desc="📊 Evaluating",
                bar_format="{l_bar}%s{bar}%s{r_bar}"
                % ("\033[" + self._get_color_code(self.eval_color), "\033[0m"),
                dynamic_ncols=True,
                leave=False,
            )
            self.last_evaluated_step = state.global_step

        # Since we don't know the evaluation size, just set to 100%
        if self.evaluation_bar is not None:
            self.evaluation_bar.n = 100
            self.evaluation_bar.refresh()
            self.evaluation_bar.close()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Close progress bars at the end of training."""
        if self.training_bar is not None:
            self.training_bar.close()
            logging.info("✨ Training Completed! ✨")

        if self.evaluation_bar is not None and not self.evaluation_bar.disable:
            self.evaluation_bar.close()

    def _get_color_code(self, color_name):
        """Convert color name to ANSI color code."""
        color_codes = {
            "black": "30m",
            "red": "31m",
            "green": "32m",
            "yellow": "33m",
            "blue": "34m",
            "magenta": "35m",
            "cyan": "36m",
            "white": "37m",
            "bright_black": "90m",
            "bright_red": "91m",
            "bright_green": "92m",
            "bright_yellow": "93m",
            "bright_blue": "94m",
            "bright_magenta": "95m",
            "bright_cyan": "96m",
            "bright_white": "97m",
        }
        return color_codes.get(color_name, "32m")  # Default to green if color not found


class BaselineCallback(TrainerCallback):
    def __init__(self, baseline_metric_name, baseline_metric_value):
        self.baseline_metric_name = baseline_metric_name
        self.baseline_metric_value = baseline_metric_value
        self.baseline_improved = False
        logging.info(
            f"Initialized baseline {baseline_metric_name} = {baseline_metric_value}"
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            logging.warning("No metrics provided during evaluation")
            return
        metric_key = f"eval_{self.baseline_metric_name}"
        if metric_key not in metrics:
            logging.warning(f"Metric {metric_key} not found in evaluation metrics")
            return
        current_metric_value = metrics[metric_key]
        logging.info(
            f"Current {self.baseline_metric_name} = {current_metric_value}, Baseline = {self.baseline_metric_value}"
        )
        if current_metric_value > self.baseline_metric_value:
            self.baseline_metric_value = current_metric_value
            self.baseline_improved = True
            control.should_save = True
            logging.info("Metric improved, saving checkpoint")
        else:
            control.should_save = False
            logging.info("Metric not improved, skipping save")

    def get_baseline_improved(self):
        return self.baseline_improved


class PlotLossCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.eval_loss = []
        self.eval_metric = []
        self.train_steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_local_process_zero:
            # Collect training loss
            if "loss" in logs:
                self.train_loss.append(logs["loss"])
                self.train_steps.append(state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and state.is_local_process_zero:
            # Collect evaluation loss
            if "eval_loss" in metrics:
                self.eval_loss.append(metrics["eval_loss"])
                self.eval_steps.append(state.global_step)
            # Collect evaluation metric specified by metric_for_best_model
            metric_key = f"eval_{args.metric_for_best_model}"
            if metric_key in metrics:
                self.eval_metric.append(metrics[metric_key])

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            # Plot 1: Training Loss
            if self.train_loss:
                plt.figure(figsize=(10, 6))
                plt.plot(self.train_steps, self.train_loss, "b-", label="Training Loss")
                plt.title("Training Loss over Steps")
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.grid(True)
                plt.legend()
                train_loss_path = os.path.join(args.output_dir, "train_loss_plot.png")
                plt.savefig(train_loss_path)
                plt.close()
                logging.info(f"📊 Training loss plot saved as '{train_loss_path}'")

            # Plot 2: Evaluation Loss
            if self.eval_loss:
                plt.figure(figsize=(10, 6))
                plt.plot(self.eval_steps, self.eval_loss, "r-", label="Evaluation Loss")
                plt.title("Evaluation Loss over Steps")
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.grid(True)
                plt.legend()
                eval_loss_path = os.path.join(args.output_dir, "eval_loss_plot.png")
                plt.savefig(eval_loss_path)
                plt.close()
                logging.info(f"📊 Evaluation loss plot saved as '{eval_loss_path}'")

            # Plot 3: Evaluation Metric (metric_for_best_model)
            if self.eval_metric and args.metric_for_best_model:
                plt.figure(figsize=(10, 6))
                plt.plot(
                    self.eval_steps,
                    self.eval_metric,
                    "g-",
                    label=f"Evaluation {args.metric_for_best_model}",
                )
                plt.title(f"Evaluation {args.metric_for_best_model} over Steps")
                plt.xlabel("Steps")
                plt.ylabel(args.metric_for_best_model)
                plt.grid(True)
                plt.legend()
                eval_metric_path = os.path.join(
                    args.output_dir, f"eval_{args.metric_for_best_model}_plot.png"
                )
                plt.savefig(eval_metric_path)
                plt.close()
                logging.info(f"📊 Evaluation metric plot saved as '{eval_metric_path}'")


class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Separate 'text' and other features
        text_features = (
            [f.pop("text") for f in features] if "text" in features[0] else None
        )
        labels_features = (
            [f.pop("labels") for f in features] if "labels" in features[0] else None
        )

        # Let DataCollatorWithPadding handle the rest
        batch = super().__call__(features)

        # Optionally, add 'text' and 'labels' back to the batch if needed
        if text_features is not None:
            batch["text"] = text_features
        if labels_features is not None:
            batch["labels"] = labels_features

        return batch


# Custom Trainer
class CustomTrainer(Trainer):
    """
    Custom Trainer to handle LambdaRank loss and token ranking visualization.
    """

    def __init__(
        self,
        *args,
        df_toks: pd.DataFrame = None,
        df_lbs: pd.DataFrame = None,
        generate_report=False,
        tokenizer: AutoTokenizer = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.df_toks = df_toks
        self.df_lbs = df_lbs
        if tokenizer:
            self.tokenizer = tokenizer
            logging.info(
                f"We are registering the tokenizer {tokenizer.name_or_path} in Custom Trainer"
            )
        self.generate_report = generate_report

    def evaluate_old(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
    ):
        """
        Override evaluate to process in batches and avoid GPU memory overload.
        Also, to include token ranking visualization and save report using batch data.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("No eval_dataset provided.")

        # eval_dataloader_old = DataLoader(eval_dataset, batch_size=self.args.per_device_eval_batch_size, shuffle=False)
        # Use the custom collator
        # Remove 'token_type_ids' from the dataset only if it exists
        if "token_type_ids" in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns(["token_type_ids"])

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=CustomDataCollator(tokenizer=self.processing_class),
        )
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0.0
        num_samples = 0
        report_data = []

        evaluation_bar = tqdm(
            eval_dataloader,
            total=len(eval_dataloader),
            desc="📊 Evaluating",
            bar_format="{l_bar}%s{bar}%s{r_bar}" % ("\033[" + "36m", "\033[0m"),
            dynamic_ncols=True,
            leave=False,
        )

        with torch.no_grad():
            for batch in evaluation_bar:
                # batch = {k: v.to(self.args.device) for k, v in batch.items()}
                batch = {
                    k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = self.model(**batch)

                if "loss" in outputs:
                    total_loss += outputs["loss"].item() * batch["input_ids"].size(0)

                all_logits.append(outputs["logits"].cpu().numpy())
                all_labels.append(outputs["labels"].cpu().numpy())
                num_samples += batch["input_ids"].size(0)

                batch_for_ranking = {
                    "input_ids": batch["input_ids"],
                    "text": batch["text"],
                    "attention_mask": batch["attention_mask"],
                    "token_type_ids": batch.get("token_type_ids", None),
                }

                if self.generate_report:
                    token_rank_results = visual_tokenrank_per_label(
                        (outputs["logits"], outputs["labels"]),
                        batch_for_ranking,
                        self.df_toks,
                        self.df_lbs,
                        top_k=50,
                    )
                    report_data.extend(token_rank_results["report_data"])

                del outputs, batch
                torch.cuda.empty_cache()

        if self.generate_report:
            # Generate and save reports
            generate_tokenrank_report(report_data, self.args.output_dir)

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        metrics = self.compute_metrics(
            (np.concatenate(all_logits), np.concatenate(all_labels))
        )
        metrics = {
            f"{metric_key_prefix}_{key}": value for key, value in metrics.items()
        }
        metrics[f"{metric_key_prefix}_loss"] = avg_loss

        # Manually trigger the on_evaluate callback
        if hasattr(self, "callback_handler"):
            self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics=metrics
            )

        self.log(metrics)
        return metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to process in batches, compute metrics incrementally, and handle multiple metrics generically.
        Includes token ranking visualization and conditional cache clearing.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("No eval_dataset provided.")
        if len(eval_dataset) == 0:
            logging.warning("Evaluation dataset is empty.")
            return {f"{metric_key_prefix}_loss": 0.0}

        # Remove 'token_type_ids' from the dataset if it exists
        if "token_type_ids" in eval_dataset.column_names:
            logging.info(
                "Removing 'token_type_ids' from eval_dataset as they are not needed."
            )
            eval_dataset = eval_dataset.remove_columns(["token_type_ids"])

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=CustomDataCollator(tokenizer=self.processing_class),
        )
        # "candidate" mode's logits/labels carry real (non-padding) slots
        # mixed with padding slots that still gather a real (non-zero)
        # label's relevance data -- compute_metrics needs candidate_mask to
        # tell them apart (see compute_metrics's row_mask docstring).
        is_candidate_mode = "candidate_mask" in eval_dataset.column_names
        logging.debug(
            f"🔎 [diagnostic] is_candidate_mode={is_candidate_mode}, "
            f"eval_dataset.column_names={eval_dataset.column_names}"
        )
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        metrics_sums = {}  # Store sum of each metric weighted by batch size
        valid_samples_per_metric = {}  # Track valid samples per metric
        report_data = []
        diag_total_rows = 0
        diag_real_rows = 0

        evaluation_bar = tqdm(
            eval_dataloader,
            total=len(eval_dataloader),
            desc="📊 Evaluating",
            bar_format="{l_bar}%s{bar}%s{r_bar}" % ("\033[36m", "\033[0m"),
            dynamic_ncols=True,
            leave=False,
        )

        with torch.no_grad():
            for batch in evaluation_bar:
                batch = {
                    k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = self.model(**batch)

                if "loss" in outputs:
                    total_loss += outputs["loss"].item() * batch["input_ids"].size(0)

                batch_size = batch["input_ids"].size(0)
                num_samples += batch_size

                # Compute batch-level metrics
                batch_logits = outputs["logits"].cpu().numpy()
                batch_labels = outputs["labels"].cpu().numpy()
                batch_row_mask = batch["candidate_mask"].cpu().numpy() if is_candidate_mode else None
                if batch_row_mask is not None:
                    diag_total_rows += batch_row_mask.size
                    diag_real_rows += int(batch_row_mask.sum())
                batch_metrics = self.compute_metrics((batch_logits, batch_labels), row_mask=batch_row_mask)

                # Accumulate metrics generically
                for metric_name, metric_value in batch_metrics.items():
                    if metric_name not in metrics_sums:
                        metrics_sums[metric_name] = 0.0
                        valid_samples_per_metric[metric_name] = 0
                    # Only accumulate if metric is valid (non-zero or non-NaN)
                    if metric_value > 0 and not np.isnan(metric_value):
                        metrics_sums[metric_name] += metric_value * batch_size
                        valid_samples_per_metric[metric_name] += batch_size

                batch_for_ranking = {
                    "input_ids": batch["input_ids"],
                    "text": batch["text"],
                    "labels": batch["labels"],
                    "attention_mask": batch["attention_mask"],
                    "token_type_ids": batch.get("token_type_ids", None),
                }

                if self.generate_report:
                    token_rank_results = visual_tokenrank_per_label(
                        (outputs["logits"], outputs["labels"]),
                        batch_for_ranking,
                        self.df_toks,
                        self.df_lbs,
                        top_k=50,
                    )
                    report_data.extend(token_rank_results["report_data"])

                # Conditional GPU cache clearing
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    total_memory = (
                        torch.cuda.get_device_properties(0).total_memory / 1024**3
                    )  # GB
                    if memory_allocated > 0.8 * total_memory:
                        torch.cuda.empty_cache()
                        logging.debug(
                            f"Cleared GPU cache: {memory_allocated:.2f}/{total_memory:.2f} GB allocated"
                        )

                del outputs, batch, batch_logits, batch_labels
                torch.cuda.empty_cache()  # Optional: keep as fallback

        if self.generate_report:
            generate_tokenrank_report(report_data, self.args.output_dir)

        if is_candidate_mode:
            logging.info(
                f"🔎 [diagnostic] candidate_mask real-vs-padding rows across eval: "
                f"{diag_real_rows}/{diag_total_rows} real "
                f"({100.0 * diag_real_rows / diag_total_rows:.1f}%)"
                if diag_total_rows > 0 else "🔎 [diagnostic] no eval rows seen"
            )

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        metrics = {f"{metric_key_prefix}_loss": avg_loss}

        # Compute average for each metric
        for metric_name in metrics_sums:
            valid_samples = valid_samples_per_metric[metric_name]
            if valid_samples > 0:
                metrics[f"{metric_key_prefix}_{metric_name}"] = (
                    metrics_sums[metric_name] / valid_samples
                )
            else:
                metrics[f"{metric_key_prefix}_{metric_name}"] = 0.0
                logging.warning(f"No valid samples found for metric '{metric_name}'.")

        # Manually trigger the on_evaluate callback
        if hasattr(self, "callback_handler"):
            self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics=metrics
            )

        self.log(metrics)
        return metrics

    def _save(self, output_dir=None, state_dict=None):
        # Use the provided output_dir or default to current checkpoint dir
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save the model weights, format gated on self.args.save_safetensors.
        # This override bypasses Trainer's own save entirely (that's the
        # point of overriding _save), so it must honor the flag itself --
        # setting save_safetensors=False on TrainingArguments alone did
        # nothing here. state_dict includes ground_model's params (it's a
        # submodule of LTRModel), and safetensors can't serialize
        # Phi-3-small's tied embed_tokens/lm_head (tie_word_embeddings=
        # True, unlike Mistral/Llama); torch.save handles aliased tensors
        # natively.
        if state_dict is None:
            state_dict = self.model.state_dict()
        if getattr(self.args, "save_safetensors", True):
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            logging.info(f"💾 Model weights saved in safetensors format: {output_dir}")
        else:
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
            logging.info(f"💾 Model weights saved in pytorch_model.bin format: {output_dir}")

        # Explicitly save the config
        if hasattr(self.model, "config") and self.model.config is not None:
            self.model.config.save_pretrained(output_dir)
            logging.info(f"⚙️ Config saved in checkpoint: {output_dir}")
        else:
            logging.warning(f"⚠️ Model has no config to save in checkpoint")

        # Save training args
        self._save_training_args(output_dir)

        # Log saved files in a tabular format with file sizes (including files in subdirectories)
        saved_files = []
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                full_path = os.path.join(root, f)
                # Get relative path from output_dir for cleaner display
                relative_path = os.path.relpath(full_path, output_dir)
                file_size = os.path.getsize(full_path) / 1024**2
                saved_files.append((relative_path, file_size)) 

        table_data = [(i + 1, f, f"{s:.2f} MB") for i, (f, s) in enumerate(saved_files)]
        table = tabulate(
            table_data, headers=["Index", "Saved File", "Size"], tablefmt="grid"
        )
        logging.info(f"📋 Saved files in {output_dir}:\n{table}")

    def _save_training_args(self, output_dir):
        # Helper to save training_args.bin
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_best_model_old(self):
        """
        Override to skip loading if baseline wasn’t improved.
        """
        # Find BaselineCallback in callbacks
        baseline_improved = False
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, BaselineCallback):
                baseline_improved = callback.get_baseline_improved()
                break

        if not self.args.load_best_model_at_end:
            logging.info("load_best_model_at_end is False, skipping best model load")
            return

        # if not baseline_improved:
        #     logging.info("Baseline was not improved during training, keeping original model")
        #     return  # Skip loading, keep the original model

        # Proceed with default loading logic if baseline improved
        import ipdb

        ipdb.set_trace()
        logging.info(f"Loading best model from {self.state.best_model_checkpoint}")
        if self.state.best_model_checkpoint and os.path.exists(
            self.state.best_model_checkpoint
        ):
            weights_file = os.path.join(
                self.state.best_model_checkpoint, "model.safetensors"
            )
            if os.path.isfile(weights_file):
                state_dict = load_file(weights_file)
                import ipdb

                ipdb.set_trace()
                self.model.load_state_dict(state_dict)
                logging.info(f"Loaded best model weights from {weights_file}")
            else:
                logging.warning(
                    f"No model.safetensors found in {self.state.best_model_checkpoint}"
                )
        else:
            logging.warning("No best checkpoint available to load")

    def _load_best_model(self):

        if not self.args.load_best_model_at_end:
            logging.info("load_best_model_at_end is False, skipping best model load")
            return

        logging.info(f"📂 Loading best model from {self.state.best_model_checkpoint}")
        if self.state.best_model_checkpoint and os.path.exists(
            self.state.best_model_checkpoint
        ):
            # _save (above) writes pytorch_model.bin instead of
            # model.safetensors whenever save_safetensors=False (needed for
            # Phi-3-small's tied embed_tokens/lm_head, which safetensors
            # can't serialize) -- so the reload must check for both.
            safetensors_file = os.path.join(
                self.state.best_model_checkpoint, "model.safetensors"
            )
            bin_file = os.path.join(
                self.state.best_model_checkpoint, "pytorch_model.bin"
            )
            weights_file = (
                safetensors_file if os.path.isfile(safetensors_file) else bin_file
            )
            if os.path.isfile(weights_file):
                # Determine the device of the model
                model_device = next(self.model.parameters()).device
                logging.info(f"🖥️ Model is on device: {model_device}")

                # Load the state_dict (tensors are typically on CPU)
                state_dict = (
                    load_file(weights_file)
                    if weights_file == safetensors_file
                    else torch.load(weights_file, map_location="cpu")
                )

                # Verify that the keys match (ignoring order)
                model_state_dict = self.model.state_dict()
                if set(state_dict.keys()) != set(model_state_dict.keys()):
                    logging.warning(
                        f"⚠️ Key mismatch: saved keys {set(state_dict.keys())} != model keys {set(model_state_dict.keys())}"
                    )
                    raise ValueError(
                        "State dict keys do not match model's state dict keys"
                    )

                # Log key order for debugging
                saved_keys = list(state_dict.keys())[:5]
                model_keys = list(model_state_dict.keys())[:5]
                table_data = list(zip(range(1, 6), saved_keys, model_keys))
                table = tabulate(
                    table_data,
                    headers=["Index", "Saved state_dict Keys", "Model state_dict Keys"],
                    tablefmt="grid",
                )
                logging.info(f"🔑 Key order comparison:\n{table}")

                # Create a new state_dict with the model's key order and move tensors to the model's device
                ordered_state_dict = {}
                for k in model_state_dict.keys():
                    if k in state_dict:
                        ordered_state_dict[k] = state_dict[k].to(
                            model_device
                        )  # Move tensor to model's device
                    else:
                        logging.warning(
                            f"⚠️ Key {k} not found in saved state_dict, skipping"
                        )

                # Load the ordered state_dict
                try:
                    self.model.load_state_dict(ordered_state_dict, strict=False)
                    logging.info(f"✅ Loaded best model weights from {weights_file}")

                    # Optional: Verify weights are loaded correctly by checking a few parameters
                    for key in list(model_state_dict.keys())[
                        :2
                    ]:  # Check first two parameters
                        if key in state_dict:
                            saved_weight = state_dict[key].to(
                                model_device
                            )  # Ensure saved weight is on same device
                            loaded_weight = self.model.state_dict()[key]
                            if torch.equal(saved_weight, loaded_weight):
                                logging.info(
                                    f"✔️ Weight for {key} matches between saved and loaded state_dict"
                                )
                            else:
                                logging.warning(f"⚠️ Weight mismatch for {key}")
                except Exception as e:
                    logging.error(f"❌ Error loading state_dict: {e}")
                    raise
            else:
                logging.warning(
                    f"⚠️ No model.safetensors or pytorch_model.bin found in {self.state.best_model_checkpoint}"
                )
        else:
            logging.warning(f"⚠️ No best checkpoint available to load")


# Function to train the model
def train_model(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    tokenizer,
    df_toks,
    df_lbs,
    k_values=[5, 8, 15],
    rare_labels=None,
    not_rare_labels=None,
    eval=False,
    generate_report=False,
    final_lr_scheduling=1e-6,
):
    """Train or evaluate the Learning-to-Rank model based on eval flag."""

    def replicated_retrieval_normalized_dcg(
        preds: Tensor,
        target: Tensor,
        top_k: Optional[int] = None,
        ignore_ties: bool = False,
    ) -> Tensor:
        """Compute Normalized Discounted Cumulative Gain (nDCG) for information retrieval.

        `preds` and `target` should be of the same shape and live on the same device.
        `target` must be either `bool` or `integers` and `preds` must be `float`,
        otherwise an error is raised.

        Args:
            preds: Estimated probabilities of each document to be relevant.
            target: Ground truth about each document relevance.
            top_k: Consider only the top k elements (default: `None`, which considers them all)
            ignore_ties: If True, ties in predictions are ignored. If False, ties are averaged (default: False)

        Return:
            A single-value tensor with the nDCG of the predictions `preds` w.r.t. the labels `target`.

        Raises:
            ValueError:
                If `top_k` parameter is not `None` or an integer larger than 0

        Example:
            >>> from torchmetrics.functional.retrieval import retrieval_normalized_dcg
            >>> preds = torch.tensor([.1, .2, .3, 4, 70])
            >>> target = torch.tensor([10, 0, 0, 1, 5])
            >>> replicated_retrieval_normalized_dcg(preds, target)
            tensor(0.6957)
            >>> replicated_retrieval_normalized_dcg(preds, target, ignore_ties=True)
            tensor(0.6957)  # May differ slightly depending on tie handling
        """
        # Validate inputs
        preds, target = _check_retrieval_functional_inputs(
            preds, target, allow_non_binary_target=True
        )
        top_k = preds.shape[-1] if top_k is None else top_k
        if not (isinstance(top_k, int) and top_k > 0):
            raise ValueError("`top_k` has to be a positive integer or None")

        # Compute discount factors
        discount = 1.0 / (
            torch.log2(torch.arange(target.shape[-1], device=target.device) + 2.0)
        )
        discount[top_k:] = 0.0

        # Compute DCG
        if ignore_ties:
            # Ignore ties: sort preds directly
            ranking = preds.argsort(descending=True)
            ranked = target[ranking]
            gain = (discount * ranked).sum()
        else:
            # Handle ties: average relevance for equal predictions
            _, inv, counts = torch.unique(
                -preds, return_inverse=True, return_counts=True
            )
            ranked = torch.zeros_like(counts, dtype=torch.float32)
            ranked.scatter_add_(0, inv, target.to(dtype=ranked.dtype))
            ranked = ranked / counts
            groups = counts.cumsum(dim=0) - 1
            discount_sums = torch.zeros_like(counts, dtype=torch.float32)
            discount_sums[0] = discount.cumsum(dim=-1)[groups[0]]
            discount_sums[1:] = discount.cumsum(dim=-1)[groups].diff()
            gain = (ranked * discount_sums).sum()

        # Compute IDCG (ideal DCG, no ties)
        ranking = target.argsort(descending=True)
        ranked_ideal = target[ranking]
        normalized_gain = (discount * ranked_ideal).sum()

        # Normalize DCG
        all_irrelevant = normalized_gain == 0
        gain = torch.where(
            all_irrelevant,
            torch.tensor(0.0, device=gain.device),
            gain / normalized_gain,
        )

        return gain.mean()

    # Function to compute metrics
    def compute_metrics(
        eval_pred: Tuple[np.ndarray, np.ndarray],
        k: int = 25,
        relevance_threshold: float = 80.0,
        row_mask: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Compute NDCG and NDCG@k with torchmetrics, replicated torchmetrics, and custom implementations.

        Args:
            eval_pred: Tuple of (predictions, labels), numpy arrays of shape (batch_size, max_len, num_labels).
                    Predictions are logits; labels are integers in [1, 101].
            k: Integer for NDCG@k (default: 25).
            row_mask: In candidate mode, shape (batch_size, num_labels) bool array
                (candidate_mask), True for real (positive or hard-negative) candidate
                slots and False for pure padding. Padding slots' candidate_label_ids
                default to a real label id (0), so their gathered relevance comes from
                that label's actual (often non-zero) lookup_table row -- NOT all-zero --
                meaning the existing non_zero_mask heuristic below cannot tell a padding
                slot from a real one on its own. Omit outside candidate mode, where every
                column is a real label and non_zero_mask alone is correct (see
                README_AMAZON3M.md's full-scale candidate-generator plan).

        Returns:
            Dict[str, float]: Dictionary containing NDCG and NDCG@k metrics.
        """
        predictions, labels = eval_pred
        # Convert to tensors and normalize predictions
        predictions = torch.sigmoid(
            torch.tensor(predictions, dtype=torch.float32)
        )  # Logits to [0, 1]
        labels = torch.tensor(labels, dtype=torch.float32)

        batch_size, max_len, num_labels = predictions.shape

        # Reshape to (batch_size * num_labels, max_len)
        predictions = predictions.permute(0, 2, 1).reshape(-1, max_len)
        labels = labels.permute(0, 2, 1).reshape(-1, max_len)

        metrics = {}
        non_zero_mask = labels.sum(dim=1) > 0
        pre_mask_count = int(non_zero_mask.sum())
        if row_mask is not None:
            # candidate_mask is already (batch_size, num_labels) with the same
            # (example, label-slot) row order the permute+reshape above produces
            # for predictions/labels -- no permute needed here, just flatten.
            non_zero_mask = non_zero_mask & torch.tensor(row_mask, dtype=torch.bool).reshape(-1)
            logging.debug(
                f"🔎 [diagnostic] compute_metrics row_mask applied: "
                f"non_zero_mask {pre_mask_count} -> {int(non_zero_mask.sum())} rows "
                f"(out of {non_zero_mask.numel()} total)"
            )

        valid_preds = predictions[non_zero_mask]
        valid_labels = labels[non_zero_mask]

        if non_zero_mask.any():
            # Torchmetrics: NDCG (full ranking)
            ndcg_scores = retrieval_normalized_dcg(valid_preds, valid_labels)
            metrics["ndcg"] = ndcg_scores.mean().item()

            # Torchmetrics: NDCG@k
            k = min(k, max_len)
            ndcg_k_scores = retrieval_normalized_dcg(valid_preds, valid_labels, top_k=k)
            metrics[f"ndcg@{k}"] = ndcg_k_scores.mean().item()

            # Torchmetrics: Precision@k
            # Convert labels to binary based on threshold for torchmetrics
            binary_labels = (valid_labels >= relevance_threshold).float()
            precision_scores = retrieval_precision(valid_preds, binary_labels, top_k=k)
            metrics[f"precision@{k}"] = precision_scores.mean().item()
        else:
            metrics["ndcg"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
            metrics[f"precision@{k}"] = 0.0

        return metrics

    # Ensure training_args is a TrainingArguments object
    if not isinstance(training_args, TrainingArguments):
        training_args = TrainingArguments(**training_args)

    # Initialize CustomTrainer
    logging.info(f"🏁 Preparing Custom Trainer 🛠️")

    callbacks = [PrettyLoggingCallback(), ColoredProgressCallback()]
    if not eval:
        callbacks.append(
            PlotLossCallback()
        )  # Add PlotLossCallback only if not in eval mode

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,  # Add the custom callbacks here
        # optimizers=optimizers
        df_toks=df_toks,
        df_lbs=df_lbs,
        generate_report=generate_report,
    )

    # If eval flag is True, skip training and simply return evaluation metrics
    if eval:
        logging.info(f"🔍 Evaluation Mode Activated ✅")
        logging.info("🚫 Skipping Training, Performing Evaluation Only 📊")

        # Performance evaluation logging
        logging.info(f"📏 Measuring Model Performance 🎯")
        eval_results = trainer.evaluate()
        return trainer

    # Compute baseline and add it with a BaselineCallback
    # baseline_metrics = trainer.evaluate()
    # baseline_metric_name = training_args.metric_for_best_model
    # baseline_metric_value = baseline_metrics[f"eval_{baseline_metric_name}"]
    # Add callback with baseline
    # baseline_callback = BaselineCallback(baseline_metric_name, baseline_metric_value)
    # trainer.add_callback(baseline_callback)

    # Train the model
    logging.info(f"🏋️ Commencing Model Training 💪")
    trainer.train()

    # Check if baseline was improved and adjust loading behavior
    # if training_args.load_best_model_at_end and not baseline_callback.get_baseline_improved():
    #     logging.info("No improvement over baseline during training. Skipping load_best_model_at_end to keep original model.")
    #     trainer.state.best_model_checkpoint = None  # Prevent loading non-existent checkpoint
    #     trainer.model = model  # Explicitly revert to original model (though not strictly needed)

    return trainer


def save_and_push(
    trainer,
    tokenizer,
    output_dir,
    logfile_path,
    commit_message="Trained L2R token ranking model on MIMIC-IV",
):
    """
    Save model and tokenizer locally, verify best checkpoint weights, and push to Hugging Face Hub.

    Args:
        trainer: CustomTrainer instance with model and training arguments.
        tokenizer: Hugging Face tokenizer to save and push.
        output_dir (str): Local directory to save model and tokenizer.
        logfile_path (str): Path to the log file to be saved alongside model files.
        commit_message (str): Commit message for the Hub push.
    """
    try:
        import shutil

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"📁 Creating/using output directory: {output_dir}")

        # Save model using CustomTrainer's save method (saves model.safetensors, config, training_args)
        trainer.save_model(output_dir)
        logging.info(f"💾 Model saved to: {output_dir}")

        # Explicitly save model with config to ensure custom attributes are preserved.
        # safe_serialization=False: see the TrainingArguments comment above --
        # Phi-3-small ties embed_tokens/lm_head, which safetensors can't
        # serialize.
        trainer.model.save_pretrained(output_dir, safe_serialization=False)
        # Ground Model was saved by the overwriiten `save_pretrained` 
        ground_model_dir = os.path.join(output_dir, "ground_model")
        logging.info(f"💾 Model and config explicitly saved to: {output_dir}")

        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        logging.info(f"🖌️ Tokenizer saved to: {output_dir}")

        # Save lookup_table
        lookup_table_path = os.path.join(output_dir, "lookup_table.pt")
        torch.save(trainer.model.lookup_table, lookup_table_path)
        logging.info(f"📊 Lookup table saved to: {lookup_table_path}")

        # Copy log file into the output directory
        if os.path.isfile(logfile_path):
            log_dst_path = os.path.join(output_dir, os.path.basename(logfile_path))
            shutil.copy2(logfile_path, log_dst_path)
            logging.info(f"📜 Log file saved to: {log_dst_path}")
        else:
            logging.warning(f"⚠️ Provided log file not found: {logfile_path}")

        # Log saved files in a tabular format with file sizes (including files in subdirectories)
        saved_files = []
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                full_path = os.path.join(root, f)
                # Get relative path from output_dir for cleaner display
                relative_path = os.path.relpath(full_path, output_dir)
                file_size = os.path.getsize(full_path) / 1024**2
                saved_files.append((relative_path, file_size)) 
        if saved_files:
            table_data = [
                (i + 1, f, f"{s:.2f} MB") for i, (f, s) in enumerate(saved_files)
            ]
            table = tabulate(
                table_data, headers=["Index", "Saved File", "Size"], tablefmt="grid"
            )
            logging.info(f"📋 Saved files in {output_dir}:\n{table}")
        else:
            logging.warning(f"⚠️ No files found in {output_dir}")

        # Verify critical files/directories before pushing. Weights: either
        # model.safetensors or pytorch_model.bin -- CustomTrainer._save
        # writes the latter whenever save_safetensors=False (Phi-3-small's
        # tied embed_tokens/lm_head can't go through safetensors).
        required_files = ["config.json", "lookup_table.pt", "ground_model"]
        missing_files = []

        has_weights_file = os.path.isfile(
            os.path.join(output_dir, "model.safetensors")
        ) or os.path.isfile(os.path.join(output_dir, "pytorch_model.bin"))
        if not has_weights_file:
            missing_files.append("model.safetensors/pytorch_model.bin")

        for item in required_files:
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path):
                # It's a file, it exists
                continue
            elif os.path.isdir(item_path):
                # It's a directory, check if it's not empty
                if not os.listdir(item_path):
                    missing_files.append(f"{item} (empty directory)")
            else:
                # Doesn't exist as file or directory
                missing_files.append(item)

        if missing_files:
            logging.error(f"❌ Missing or empty required files/directories: {missing_files}")
            raise FileNotFoundError(f"Missing or empty required files/directories: {missing_files}")

        # Verify weights from best checkpoint match the model's current state_dict
        if trainer.state.best_model_checkpoint:
            best_checkpoint_safetensors_file = os.path.join(
                trainer.state.best_model_checkpoint, "model.safetensors"
            )
            best_checkpoint_bin_file = os.path.join(
                trainer.state.best_model_checkpoint, "pytorch_model.bin"
            )
            best_checkpoint_weights_file = (
                best_checkpoint_safetensors_file
                if os.path.isfile(best_checkpoint_safetensors_file)
                else best_checkpoint_bin_file
            )
            if os.path.isfile(best_checkpoint_weights_file):
                logging.info(
                    f"🔍 Verifying weights from best checkpoint: {best_checkpoint_weights_file}"
                )
                best_state_dict = (
                    load_file(best_checkpoint_weights_file)
                    if best_checkpoint_weights_file == best_checkpoint_safetensors_file
                    else torch.load(best_checkpoint_weights_file, map_location="cpu")
                )
                model_device = next(trainer.model.parameters()).device
                model_state_dict = trainer.model.state_dict()

                # Ensure best_state_dict tensors are on the same device
                best_state_dict = {
                    k: v.to(model_device) for k, v in best_state_dict.items()
                }

                # Compare keys and weights
                if set(best_state_dict.keys()) != set(model_state_dict.keys()):
                    logging.warning(
                        f"⚠️ Key mismatch between best checkpoint and model: {set(best_state_dict.keys())} != {set(model_state_dict.keys())}"
                    )
                else:
                    weights_match = True
                    for key in list(model_state_dict.keys())[
                        :2
                    ]:  # Check first two parameters
                        if key in best_state_dict and not torch.equal(
                            best_state_dict[key], model_state_dict[key]
                        ):
                            logging.warning(
                                f"⚠️ Weight mismatch for {key} in best checkpoint"
                            )
                            weights_match = False
                    if weights_match:
                        logging.info(
                            f"✔️ Weights from best checkpoint match model's current state_dict"
                        )
                    else:
                        logging.warning(
                            f"⚠️ Some weights from best checkpoint do not match model's current state_dict"
                        )
            else:
                logging.warning(
                    f"⚠️ No model.safetensors or pytorch_model.bin found in best checkpoint: {trainer.state.best_model_checkpoint}"
                )
        else:
            logging.warning(f"⚠️ No best checkpoint available to verify weights")

        # Validate Hub credentials
        if trainer.args.push_to_hub:
            try:
                user_info = whoami(token=trainer.args.hub_token)
                logging.info(
                    f"🔐 Hub credentials validated for user: {user_info['name']}"
                )
            except Exception as e:
                logging.error(f"❌ Invalid Hub credentials: {e}")
                raise ValueError(f"Invalid Hub credentials: {e}")

            # Push to Hugging Face Hub
            logging.info(
                f"🚀 Pushing model to Hugging Face Hub: {trainer.args.hub_model_id}"
            )
            trainer.push_to_hub(commit_message=commit_message)
            logging.info(f"✅ Model pushed to Hub: {trainer.args.hub_model_id}")

            logging.info(
                f"🚀 Pushing tokenizer to Hugging Face Hub: {trainer.args.hub_model_id}"
            )
            tokenizer.push_to_hub(
                trainer.args.hub_model_id, token=trainer.args.hub_token
            )
            logging.info(f"✅ Tokenizer pushed to Hub: {trainer.args.hub_model_id}")

            # Push lookup_table manually using HF Hub API
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=lookup_table_path,
                path_in_repo="lookup_table.pt",
                repo_id=trainer.args.hub_model_id,
                token=trainer.args.hub_token,
                repo_type="model",
            )
            logging.info("✅ Lookup table pushed to Hub.")

            # Push ground_model dir manually using HF Hub API
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=ground_model_dir,
                path_in_repo="ground_model",
                repo_id=trainer.args.hub_model_id,
                token=trainer.args.hub_token,
                repo_type="model",
            )
            logging.info("✅ Ground Model directory pushed to Hub.")


            # Push log file manually using HF Hub API
            if os.path.isfile(log_dst_path):
                from huggingface_hub import HfApi

                api = HfApi()
                api.upload_file(
                    path_or_fileobj=log_dst_path,
                    path_in_repo=os.path.basename(logfile_path),
                    repo_id=trainer.args.hub_model_id,
                    token=trainer.args.hub_token,
                    repo_type="model",
                )
                logging.info("✅ Log file pushed to Hub.")
        else:
            logging.warning(
                f"⚠️ push_to_hub is disabled in TrainingArguments, skipping Hub push"
            )

    except Exception as e:
        logging.error(f"❌ Error during save and push: {e}")
        raise


def cleanup_gpu_memory(accelerator=None):
    """
    Clean up GPU memory and training objects from device 0 before evaluation.

    Args:
        accelerator (Accelerator, optional): The Accelerator object to free memory from, if available.
    """
    logging.info("Cleaning up GPU memory before evaluation...")

    # Delete training-related objects if they exist in locals
    for var_name in [
        "trainer",
        "model",
        "tokenized_dataset",
        "train_dataset",
        "eval_dataset",
        "data_collator",
    ]:
        if var_name in locals():
            del locals()[var_name]

    # Free Accelerator memory if provided
    if accelerator is not None:
        accelerator.free_memory()
        del accelerator

    # Clear GPU cache
    torch.cuda.empty_cache()

    logging.info("GPU memory cleaned up.")


def load_file_from_hub(hub_model_id, filename):
    """Load a file from the Hub using HfApi."""
    api = HfApi()
    file_path = api.hf_hub_download(repo_id=hub_model_id, filename=filename)
    return load_file(file_path)


def load_custom_task_model_and_tokenizer_old(
    output_dir,
    hub_model_id,
    task_hub_model_id,
    model_exists_locally,
    do_fresh_training,
    device="cpu",
    hf_token=None,
    **kwargs,
):
    """
    Load the model and tokenizer from either a local path or the Hub, handling custom config.

    Args:
        model_path (str): Path to load from (will be determined).
        output_dir (str): Local directory for saving/loading.
        hub_model_id (str): Hugging Face Hub model ID.
        model_exists_locally (bool): Whether a local model exists.
        do_fresh_training (bool): Whether to force a fresh training run.
        device (str): Device to load model onto (default: "cpu").

    Returns:
        tuple: (model, tokenizer)
    """
    # Check if model exists locally or on Hub (unless do_fresh_training is True)
    model_exists_locally = os.path.exists(os.path.join(output_dir, "config.json"))

    # Determine where to load the model from
    if model_exists_locally and not do_fresh_training:
        model_path = output_dir
        logging.info(f"Loading model from local path: {model_path}")
    else:
        model_path = task_hub_model_id
        logging.info(f"Loading model from Hub: {model_path}")

    # Load config manually as a dictionary
    config_path = (
        os.path.join(model_path, "config.json") if model_exists_locally else None
    )
    if model_exists_locally and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        logging.info(f"Loaded config from local path: {config_path}")
    else:
        api = HfApi()
        config_file = api.hf_hub_download(repo_id=model_path, filename="config.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        logging.info(f"Loaded config from Hub: {model_path}")

    # Verify expected model type
    # if config_dict.get("model_type") != "multilabel_icd_classifier":
    #     logging.warning(f"Config model_type is {config_dict.get('model_type')}, expected 'multilabel_icd_classifier'")

    # Extract key config values
    num_labels = config_dict.get("num_labels")
    base_model_path = config_dict.get(
        "_name_or_path", "mistralai/Mistral-7B-Instruct-v0.3"
    )  # Default if missing
    if not num_labels:
        logging.error("Config missing 'num_labels' field")
        raise ValueError("Config must specify 'num_labels'")

    # Load the domain-adapted LLM first
    # base_model, _ = load_base_model_and_tokenizer(base_model_path, hub_model_id, task_hub_model_id,  hf_token)
    base_model, _ = load_base_model_and_tokenizer(base_model_path)

    # Initialize custom model with the multi-label classification head
    if "lookup_table" not in kwargs:
        raise ValueError("lookup_table not found in kwargs")
    lookup_table = kwargs["lookup_table"]
    model = define_model(base_model, num_labels, lookup_table)
    logging.info(f"Initialized L2R model with {num_labels} labels")

    # Load custom weights that were trained
    safetensors_path = (
        os.path.join(output_dir, "model.safetensors")
        if model_exists_locally
        else model_path
    )
    try:
        custom_state_dict = (
            load_file(safetensors_path)
            if model_exists_locally
            else load_file_from_hub(model_path, "model.safetensors")
        )
        model.load_state_dict(custom_state_dict, strict=False)
        ###
        # I WILL SPECIFY LATER
        ###
        logging.info("Loaded custom weights from model.safetensors")
    except Exception as e:
        logging.error(f"Failed to load custom weights from {safetensors_path}: {e}")
        raise

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logging.info(f"Loaded tokenizer from: {model_path}")
    except Exception as e:
        logging.error(f"Failed to load tokenizer from {model_path}: {e}")
        raise

    # Move model to specified device
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")

    return model, tokenizer


def load_custom_task_model_and_tokenizer(
    output_dir,
    task_hub_model_id,
    model_exists_locally,
    do_fresh_training,
    device="cpu",
    hf_token=None,
    skip_push_to_hub=False,
    **kwargs,
):
    """
    Load the model and tokenizer from either a local path or the Hub, handling custom config.

    Args:
        output_dir (str): Local directory for saving/loading.
        hub_model_id (str): Hugging Face Hub model ID for base model.
        task_hub_model_id (str): Hugging Face Hub model ID for custom task model.
        model_exists_locally (bool): Whether a local model exists.
        do_fresh_training (bool): Whether to force a fresh training run.
        device (str): Device to load model onto (default: "cpu").
        hf_token (str, optional): Hugging Face Hub token for authentication.
        **kwargs: Additional arguments, including 'lookup_table' for model initialization.

    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # Check if model exists locally
        model_exists_locally = os.path.exists(os.path.join(output_dir, "config.json"))
        logging.info(f"🔍 Model exists locally: {model_exists_locally}")

        # Determine where to load the model from. skip_push_to_hub forces the
        # local path unconditionally, since no Hub copy exists to fall back to.
        if skip_push_to_hub or (model_exists_locally and not do_fresh_training):
            model_path = output_dir
            logging.info(f"📂 Loading model from local path: {model_path}")
        else:
            model_path = task_hub_model_id
            logging.info(f"🌐 Loading model from Hub: {model_path}")

        # Load config manually as a dictionary
        config_dict = None
        config_path = (
            os.path.join(model_path, "config.json") if model_exists_locally else None
        )
        if model_exists_locally and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            logging.info(f"⚙️ Loaded config from local path: {config_path}")
        else:
            api = HfApi(token=hf_token)
            config_file = api.hf_hub_download(
                repo_id=model_path, filename="config.json", token=hf_token
            )
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            logging.info(f"🌐 Loaded config from Hub: {model_path}")

        # Verify expected model type (optional, commented out as per original)
        # if config_dict.get("model_type") != "multilabel_icd_classifier":
        #     logging.warning(f"⚠️ Config model_type is {config_dict.get('model_type')}, expected 'multilabel_icd_classifier'")

        # Extract key config values
        num_labels = config_dict.get("num_labels")
        label_embed_rank = config_dict.get("label_embed_rank")
        base_model_path = config_dict.get(
            "_name_or_path", "mistralai/Mistral-7B-Instruct-v0.3"
        )
        if not num_labels:
            logging.error(f"❌ Config missing 'num_labels' field")
            raise ValueError("Config must specify 'num_labels'")
        logging.info(f"🔢 Num labels: {num_labels}, Base model path: {base_model_path}")

        # Load the domain-adapted base model
        base_model, _ = load_base_model_and_tokenizer(base_model_path)
        logging.info(f"🤖 Loaded base model from: {base_model_path}")

        # Initialize custom model with the multi-label classification head
        if "lookup_table" not in kwargs:
            logging.error(f"❌ lookup_table not found in kwargs")
            raise ValueError("lookup_table not found in kwargs")
        lookup_table = kwargs["lookup_table"]

        logging.info("Before preparing the model the % of trainable params are:")
        log_print_output(compute_trainable_params, base_model)
        # label_embed_rank must match how this checkpoint was originally
        # trained -- read from its own saved config rather than the CLI, so a
        # factorized checkpoint always reloads with the same architecture it
        # was saved with (mirrors multilabel_classify.py's
        # load_custom_task_model_and_tokenizer).
        model = define_model(base_model, num_labels, lookup_table, label_embed_rank=label_embed_rank)
        logging.info(f"🏗️ Initialized L2R model with {num_labels} labels")
        logging.info("After preparing the model the % of trainable params are:")
        log_print_output(func=model.compute_trainable_params)

        # Move model to specified device
        model = model.to(device)
        logging.info(f"🖥️ Model moved to device: {device}")

        # Load custom weights
        safetensors_path = (
            os.path.join(model_path, "model.safetensors")
            if model_exists_locally
            else None
        )
        try:
            if model_exists_locally and os.path.exists(safetensors_path):
                custom_state_dict = load_file(safetensors_path)
                source = "local path"
                files_dir = model_path  # Use model_path for local loading
            else:
                api = HfApi(token=hf_token)
                safetensors_path = api.hf_hub_download(
                    repo_id=model_path, filename="model.safetensors", token=hf_token
                )
                custom_state_dict = load_file(safetensors_path)
                source = "Hub"
                files_dir = os.path.dirname(
                    safetensors_path
                )  # Use parent directory of safetensors_path for Hub loading
            logging.info(f"💾 Loaded custom weights from {source}: {safetensors_path}")

            # Move tensors to the specified device
            custom_state_dict = {k: v.to(device) for k, v in custom_state_dict.items()}

            # Align state_dict keys with model's expected order
            model_state_dict = model.state_dict()
            if set(custom_state_dict.keys()) != set(model_state_dict.keys()):
                logging.warning(
                    f"⚠️ Key mismatch: loaded keys {set(custom_state_dict.keys())} != model keys {set(model_state_dict.keys())}"
                )
            ordered_state_dict = {
                k: custom_state_dict[k]
                for k in model_state_dict.keys()
                if k in custom_state_dict
            }

            # Load weights with non-strict mode
            model.load_state_dict(ordered_state_dict, strict=False)
            logging.info(f"✅ Custom weights loaded into model")

            # Verify weights match
            weights_match = True
            for key in list(model_state_dict.keys())[:2]:  # Check first two parameters
                if key in custom_state_dict and not torch.equal(
                    custom_state_dict[key], model_state_dict[key]
                ):
                    logging.warning(f"⚠️ Weight mismatch for {key}")
                    weights_match = False
            if weights_match:
                logging.info(f"✔️ Loaded weights match model's state_dict")
            else:
                logging.warning(
                    f"⚠️ Some loaded weights do not match model's state_dict"
                )

            # Log loaded files in a tabular format
            # loaded_files = []
            # import ipdb; ipdb.set_trace()
            # if model_exists_locally:
            #     loaded_files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f)) and f in ["config.json", "model.safetensors"]]
            # else:
            #     loaded_files = ["config.json", "model.safetensors"]
            # if loaded_files:
            #     table_data = [(i + 1, f, f"{os.path.getsize(os.path.join(model_path, f)) / 1024**2:.2f} MB" if model_exists_locally else "N/A")
            #                   for i, f in enumerate(loaded_files)]
            #     table = tabulate(table_data, headers=["Index", "Loaded File", "Size"], tablefmt="grid")
            #     logging.info(f"📋 Loaded files from {source}:\n{table}")

            # Log loaded files in a tabular format
            loaded_files = [
                f
                for f in os.listdir(files_dir)
                if os.path.isfile(os.path.join(files_dir, f))
                and f in ["config.json", "model.safetensors"]
            ]
            if loaded_files:
                table_data = [
                    (
                        i + 1,
                        f,
                        f"{os.path.getsize(os.path.join(files_dir, f)) / 1024**2:.2f} MB",
                    )
                    for i, f in enumerate(loaded_files)
                ]
                table = tabulate(
                    table_data,
                    headers=["Index", "Loaded File", "Size"],
                    tablefmt="grid",
                )
                logging.info(
                    f"📋 Loaded files from {source} (directory: {files_dir}):\n{table}"
                )
            else:
                logging.warning(f"⚠️ No relevant files found in {files_dir}")

        except Exception as e:
            logging.error(
                f"❌ Failed to load custom weights from {safetensors_path}: {e}"
            )
            raise

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, token=hf_token, trust_remote_code=True
            )
            logging.info(f"🖌️ Loaded tokenizer from: {model_path}")
        except Exception as e:
            logging.error(f"❌ Failed to load tokenizer from {model_path}: {e}")
            raise

        # Move model to specified device
        model = model.to(device)
        logging.info(f"🖥️ Model moved to device: {device}")

        return model, tokenizer

    except Exception as e:
        logging.error(f"❌ Error in load_custom_task_model_and_tokenizer: {e}")
        raise


def load_tokenizer(task_hub_model_id):
    """
    Load tokenizer for a given task hub model ID and log when it was saved in the repository.

    Args:
        task_hub_model_id (str): The ID of the model to load the tokenizer for

    Returns:
        tokenizer: The loaded tokenizer

    Raises:
        RuntimeError: If tokenizer loading fails
    """
    try:
        logging.info(f"🔤 Loading tokenizer for model: {task_hub_model_id}...")

        # First, get repository info to check when it was last updated
        try:
            from huggingface_hub import model_info

            repo_info = model_info(task_hub_model_id)
            last_modified = repo_info.lastModified
            logging.info(f"📅 Repository last modified: {last_modified}")
        except Exception as e:
            logging.warning(f"⚠️ Could not retrieve repository timestamp: {e}")

        # Now load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(task_hub_model_id, trust_remote_code=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            logging.info("📝 Setting pad token to eos token...")
            tokenizer.pad_token = tokenizer.eos_token

        logging.info(f"✅ Tokenizer loaded successfully for model: {task_hub_model_id}")
        return tokenizer
    except Exception as e:
        logging.error(f"❌ Failed to load tokenizer: {e}")
        raise RuntimeError(f"Tokenizer loading failed: {e}")


# =============================================================================
# Main Function
# =============================================================================
def main(
    output_dir: str = "task2_output",
    source_url: str = "XURLs.MIMIC4",
    data: str = "mimic4_icd10_full",
    data_l2r_fname_prefix: str = "mimic4_icd10",
    logfile: str = "classification_log",
    base_dir: str = None,
    l2r_boot_dir: str = "mimic4_l2rboot",
    hub_model_id: str = "deb101/mistral-7b-mimic4",  # Task 1 Domain Adaptation model
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    max_length: int = 512,
    do_fresh_training: bool = False,
    load_from_checkpoint: bool = False,
    task: str = "l2r",
    num_train_epochs: int = 3,
    metric_for_best_model: str = "ndcg",
    learning_rate: float = 2e-4,
    warmup_steps: int = 500,
    generate_report: bool = False,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    skip_push_to_hub: bool = False,
    label_space: str = "label",
    candidate_budget: int = 256,
    label_embed_rank: int = None,
):
    # -----------------------------------------------------------------------------
    # Data and File Setup
    # -----------------------------------------------------------------------------
    # Setup logger
    logfile_path = setup_logger(**locals())
    log_section("INITIALIZING TRAINING ENVIRONMENT", "=")

    logging.info("🚀 Setting up data paths and environment variables...")

    source = untar_xxx(eval(source_url))
    base_dir = base_dir + "/" + source_url.split(".")[1]
    output_dir = str(Path(base_dir) / output_dir)
    l2r_boot_dir = str(Path(base_dir) / l2r_boot_dir)

    # print arguments
    print_args(**locals())

    # record git repo stats
    logging.info(f"\n🚀 Quick Git Info: {get_compact_git_info()}")

    # Check if directory exists
    if not os.path.exists(l2r_boot_dir):
        logging.error(f"❌ Directory not found: {l2r_boot_dir}")
        raise FileNotFoundError(f"Directory not found: {l2r_boot_dir}")
    logging.info(f"📁 Using L2R boot directory: {l2r_boot_dir}")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"📂 Using output directory: {output_dir}")
    if label_space == "candidate" and generate_report:
        logging.warning(
            "⚠️ --generate_report with --label_space=candidate: token-rank reports "
            "look up label names via df_lbs by column position, but in candidate mode "
            "column k means candidate_label_ids[row, k] -- a different real label per "
            "row, not df_lbs's global 'lbl' id. Reports will misattribute label names; "
            "this hasn't been fixed (unlike the analogous eval-metrics remap in "
            "multilabel_classify.py, see README_AMAZON3M.md). Safe to ignore if you only "
            "need training/eval metrics, not the token-rank report itself."
        )

    log_section("LOADING DATASETS", "+")
    logging.info("📊 Loading main dataset and L2R dataset...")

    # Load datasets
    df_data, df_l2r, df_toks, df_lbs = load_datasets(
        source, l2r_boot_dir, data, data_l2r_fname_prefix
    )
    scored_tokens = quantize_l2r_data(df_l2r, qnts=None, device=None)
    test_scored_tokens(
        scored_tokens, df_toks, df_lbs, num_labels_to_sample=5, num_tokens_per_label=50
    )
    log_section("STARTING LEARNING TO RANK MODEL TRAINING", "*")

    # Hardcode your HF token (or load from environment variable)
    # hf_token = "your_hf_token_here"  # Replace or use os.environ["HF_TOKEN"]
    hf_token = os.environ["HF_TOKEN"]
    logging.info("🔐 Loaded authentication token from environment")

    # Create task specific hub model id
    task_hub_model_id = f"{hub_model_id}-{task}"
    logging.info(f"🏷️ Hub Model ID for this Learning to Rank task: {task_hub_model_id}")

    log_section("MODEL EXISTENCE CHECK", "-")

    # Check if model exists locally or on Hub
    model_exists_locally, model_exists_on_hub = check_model_existence(
        output_dir, task_hub_model_id, hf_token
    )

    # Train if model is not found locally or on Hub, or if do_fresh_training is True
    if do_fresh_training or not (model_exists_locally or model_exists_on_hub):
        try:
            log_section("STARTING FRESH TRAINING", "+")
            logging.info(
                "🔄 Starting fresh training (either forced or model not found)..."
            )
            authenticate_hf(hf_token)

            if load_from_checkpoint:
                log_section("DATA PREPROCESSING", "+")
                logging.info("🔄 Loading and preprocessing training data...")
                # Preprocess data
                tokenizer = load_tokenizer(task_hub_model_id)
                dataset, num_labels, lookup_table = preprocess_data(
                    df_data,
                    scored_tokens,
                    df_toks,
                    df_lbs,
                    tokenizer,
                    max_len=max_length,
                    batch_size=10,
                    # "cpu", not "cuda": scored_tokens (num_labels, num_tokens, 4)
                    # and the resulting lookup_table (vocab_size, num_labels) are
                    # tens of GB at Wiki10-31K full scale (30938 labels x ~128K
                    # tokens) -- eagerly placing them on GPU here OOM'd Stage 4
                    # right after the already-loaded quantized+LoRA model, before
                    # training even started. LTRModel.forward() only ever needs
                    # small per-batch (batch_size, max_len, ...) slices on GPU,
                    # and already moves those over explicitly
                    # (`.to(hidden_states.device)`) -- system RAM (525GB) easily
                    # holds the full tables, GPU (94.5GB) doesn't.
                    device="cpu",
                    save_to_disk=None,
                    label_space=label_space,
                    candidate_budget=candidate_budget,
                    base_dir=base_dir,
                )
                logging.info(f"🏷️ Number of unique ICD-10 codes: {num_labels}")

                log_section("LOADING FROM CHECKPOINT", "+")
                logging.info("📥 Loading existing model from checkpoint...")
                # Load the custom task specific model and the tokenizer
                custom_model, tokenizer = load_custom_task_model_and_tokenizer(
                    output_dir,
                    task_hub_model_id,
                    model_exists_locally,
                    do_fresh_training,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    hf_token=hf_token,
                    lookup_table=lookup_table,
                    skip_push_to_hub=skip_push_to_hub,
                )
                logging.info(
                    f"✅ Successfully loaded from checkpoint, will now continue training for {num_train_epochs} epochs."
                )
                custom_model.train()

            else:
                log_section("LOADING BASE MODEL", "+")
                logging.info("📥 Loading pretrained model and tokenizer...")
                # Load Task 1 (Domain Adapted) model and tokenizer
                # base_model, tokenizer = load_base_model_and_tokenizer(
                #     model_name, hub_model_id, hf_token
                # )
                base_model, tokenizer = load_base_model_and_tokenizer(model_name)

                log_section("DATA PREPROCESSING", "+")
                logging.info("🔄 Loading and preprocessing training data...")
                # Preprocess data
                dataset, num_labels, lookup_table = preprocess_data(
                    df_data,
                    scored_tokens,
                    df_toks,
                    df_lbs,
                    tokenizer,
                    max_len=max_length,
                    batch_size=10,
                    # "cpu", not "cuda": scored_tokens (num_labels, num_tokens, 4)
                    # and the resulting lookup_table (vocab_size, num_labels) are
                    # tens of GB at Wiki10-31K full scale (30938 labels x ~128K
                    # tokens) -- eagerly placing them on GPU here OOM'd Stage 4
                    # right after the already-loaded quantized+LoRA model, before
                    # training even started. LTRModel.forward() only ever needs
                    # small per-batch (batch_size, max_len, ...) slices on GPU,
                    # and already moves those over explicitly
                    # (`.to(hidden_states.device)`) -- system RAM (525GB) easily
                    # holds the full tables, GPU (94.5GB) doesn't.
                    device="cpu",
                    save_to_disk=None,
                    label_space=label_space,
                    candidate_budget=candidate_budget,
                    base_dir=base_dir,
                )
                logging.info(f"🏷️ Number of unique ICD-10 codes: {num_labels}")

                log_section("MODEL INITIALIZATION", "+")
                logging.info(
                    "🧠 Initializing custom L2R model for outputting per-token relevance scores per ICD-10 codes."
                )
                # Define custom model
                custom_model = define_model(base_model, num_labels, lookup_table, label_embed_rank=label_embed_rank)

            log_section("TRAINING PREPARATION", "+")
            logging.info("⚙️ Preparing training components and optimizers...")
            # Prepare model and datasets with Accelerator and also the TrainingArguments
            custom_model, train_dataset, eval_dataset, training_args, accelerator = (
                prepare_training_components(
                    custom_model,
                    dataset,
                    output_dir,
                    task_hub_model_id,
                    hf_token,
                    num_train_epochs=num_train_epochs,
                    learning_rate=learning_rate,
                    warmup_steps=warmup_steps,
                    metric_for_best_model=metric_for_best_model,
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    push_to_hub=not skip_push_to_hub,
                )
            )
            log_training_configuration(
                training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
            )

            log_section("MODEL TRAINING", "+")
            logging.info("🏋️ Starting model training process...")
            # Train
            trainer = train_model(
                custom_model,
                training_args,
                train_dataset,
                eval_dataset,
                tokenizer,
                df_toks,
                df_lbs,
                # rare_labels=rare_labels,
                # not_rare_labels=not_rare_labels,
            )

            log_section("MODEL SAVING", "+")
            logging.info("💾 Saving trained model and pushing to Hugging Face Hub...")
            # Save and push
            if accelerator.is_main_process:
                save_and_push(trainer, tokenizer, output_dir, logfile_path)

            # Done Training and Evaluation, now exit
            log_section("TRAINING COMPLETE", "=")
            logging.info("🎉 Model Training and Evaluation successfully completed.")
        except Exception as e:
            log_section("ERROR OCCURRED", "!", 80)
            logging.error(f"❌ An error occurred during model training: {e}")
            sys.exit(1)
        # No `finally: sys.exit(0)` here (removed): sys.exit() inside a
        # finally clause always wins over a pending exception/SystemExit from
        # the try/except above, so this unconditionally exited 0 regardless
        # of whether training actually succeeded -- every real failure here
        # (e.g. the CUDA OOM this comment sits next to) was silently reported
        # as success at the process level; only the launch script's separate
        # output-file-count check ever caught it. On success, just fall
        # through to the evaluation section below, same as the "skip
        # training" branch already does.
    else:
        log_section("TRAINING SKIPPED", "-")
        logging.info("⏭️ Model found locally or on Hub, skipping training...")

    # -----------------------------------------------------------------------------
    # Cleanup Before Evaluation
    # -----------------------------------------------------------------------------
    log_section("CLEANUP BEFORE EVALUATION", "-")
    logging.info("🧹 Cleaning up GPU memory before evaluation...")
    cleanup_gpu_memory(accelerator if "accelerator" in locals() else None)

    # -----------------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------------
    # Preprocess data
    log_section("PREPARING EVALUATION DATA", "+")
    logging.info("🔄 Loading and preprocessing evaluation data...")
    tokenizer = load_tokenizer(task_hub_model_id)
    dataset, num_labels, lookup_table = preprocess_data(
        df_data,
        scored_tokens,
        df_toks,
        df_lbs,
        tokenizer,
        max_len=max_length,
        batch_size=10,
        # "cpu": see the training-path preprocess_data() calls above for why
        # (scored_tokens/lookup_table are tens of GB at full scale; only
        # small per-batch slices need to reach the GPU).
        device="cpu",
        save_to_disk=None,
        label_space=label_space,
        candidate_budget=candidate_budget,
        base_dir=base_dir,
    )
    logging.info(f"🏷️ Number of unique ICD-10 codes: {num_labels}")

    log_section("STARTING MODEL EVALUATION", "*")
    logging.info("📊 Running comprehensive model evaluation...")

    # Load the custom task specific model and the tokenizer
    logging.info("📥 Loading model for evaluation...")
    custom_model, tokenizer = load_custom_task_model_and_tokenizer(
        output_dir,
        task_hub_model_id,
        model_exists_locally,
        do_fresh_training,
        device="cuda" if torch.cuda.is_available() else "cpu",
        hf_token=hf_token,
        lookup_table=lookup_table,
        skip_push_to_hub=skip_push_to_hub,
    )

    # Prepare model and datasets with Accelerator and also the TrainingArguments
    log_section("SETTING UP EVALUATION ENVIRONMENT", "+")
    logging.info("⚙️ Preparing model components for evaluation...")
    custom_model, train_dataset, eval_dataset, training_args, accelerator = (
        prepare_training_components(
            custom_model,
            dataset,
            output_dir,
            task_hub_model_id,
            hf_token,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            push_to_hub=not skip_push_to_hub,
        )
    )

    # Evaluate with train_model
    log_section("RUNNING EVALUATION", "+")
    logging.info("📏 Evaluating model performance on test dataset...")
    logging.info("📊 Using k values of 5, 8, and 15 for evaluation metrics...")
    trainer = train_model(
        custom_model,
        training_args,
        train_dataset,
        eval_dataset,
        tokenizer,
        df_toks,
        df_lbs,
        # rare_labels=rare_labels,
        # not_rare_labels=not_rare_labels,
        eval=True,
        generate_report=generate_report,
    )

    log_section("EVALUATION COMPLETE", "=")
    logging.info("🏆 Model evaluation completed successfully.")

    sys.exit(0)


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    # Convert the Namespace to a dictionary and pass to main via dictionary unpacking.
    main(**vars(args))
