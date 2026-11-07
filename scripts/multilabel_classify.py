# =============================================================================
# Imports
# =============================================================================
import warnings
from pathlib import Path
import logging
from datetime import datetime
from itertools import chain
from colorama import Fore, Style, init
import matplotlib.pyplot as plt

from tabulate import tabulate
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file
import pandas as pd
import wandb
import textwrap
import io
from contextlib import redirect_stdout

from transformers import (
    AdamW,
    get_scheduler,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    pipeline,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
from datasets import Dataset, DatasetDict
import pyarrow as pa
from accelerate import Accelerator
from huggingface_hub import HfApi, login, whoami, model_info
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from ptranking.ltr_adhoc.eval.eval_utils import LABEL_TYPE

from xcube.imports import *
from xcube.data.all import *
from xcube.l2r.models.core import LTRConfig, LTRModel, fixed_approxNDCG_loss
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


def log_with_box(message, emoji_prefix="🤖🏷️", emoji_suffix="🏷️🤖", length=80):
    """
    Creates a well-defined box around a logging message using tabulate.

    Args:
        message (str): The message to display in the box
        emoji_prefix (str): Emojis to display before the message
        emoji_suffix (str): Emojis to display after the message
        length (int): The target length of the box (default: 80)
    """
    decorated_message = f"{emoji_prefix} {message} {emoji_suffix}"

    # Calculate padding to reach the desired length
    content_length = len(decorated_message)
    if content_length < length:
        padding_needed = length - content_length
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding
        padded_message = " " * left_padding + decorated_message + " " * right_padding
    else:
        padded_message = decorated_message

    # Create a table with the message as a single cell
    table_data = [[padded_message]]

    # Use tabulate to create a box around the message
    boxed_message = tabulate(table_data, tablefmt="grid")

    # Log each line of the boxed message
    for line in boxed_message.split("\n"):
        logging.info(line)


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
        "--logfile",
        type=str,
        default="classification_log",
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
        "--metric_for_best_model",
        type=str,
        default="f1_macro",
        choices=[
            "f1_micro",
            "f1_macro",
            "precision_at_1",
            "recall_at_1",
            "precision_at_5",
            "recall_at_5",
            "precision_at_8",
            "recall_at_8",
            "precision_at_15",
            "recall_at_15",
            "precision_at_25",
            "recall_at_25",
            "rare_f1_micro",
            "rare_f1_macro",
            "rare_precision",
            "rare_recall",
            "rare_precision_at_1",
            "rare_recall_at_1",
            "rare_precision_at_5",
            "rare_recall_at_5",
            "rare_precision_at_8",
            "rare_recall_at_8",
            "rare_precision_at_15",
            "rare_recall_at_15",
            "rare_precision_at_25",
            "rare_recall_at_25",
            "not_rare_f1_micro",
            "not_rare_f1_macro",
            "not_rare_precision",
            "not_rare_recall",
            "not_rare_precision_at_1",
            "not_rare_recall_at_1",
            "not_rare_precision_at_5",
            "not_rare_recall_at_5",
            "not_rare_precision_at_8",
            "not_rare_recall_at_8",
            "not_rare_precision_at_15",
            "not_rare_recall_at_15",
            "not_rare_precision_at_25",
            "not_rare_recall_at_25",
            # train-frequency buckets: zero-shot / few-shot / head
            "zero_precision_at_1", "zero_recall_at_1", "zero_precision_at_5", "zero_recall_at_5",
            "zero_precision_at_8", "zero_recall_at_8", "zero_precision_at_15", "zero_recall_at_15",
            "zero_f1_micro", "zero_f1_macro",
            "few_precision_at_1", "few_recall_at_1", "few_precision_at_5", "few_recall_at_5",
            "few_precision_at_8", "few_recall_at_8", "few_precision_at_15", "few_recall_at_15",
            "few_f1_micro", "few_f1_macro",
            "head_precision_at_1", "head_recall_at_1", "head_precision_at_5", "head_recall_at_5",
            "head_precision_at_8", "head_recall_at_8", "head_precision_at_15", "head_recall_at_15",
            "head_f1_micro", "head_f1_macro",
        ],
        help="Metric to determine the best model (default: f1_macro).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for the optimizer (default: 2e-4)",
    )
    parser.add_argument(
        "--final_lr_scheduling",
        type=float,
        default=1e-6,
        help="Final learning rate for scheduling (default: 1e-6)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for training (default: 500)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="your-mindblowing-nlp-task-here",
        help="Task type for the model (default: your-mindblowing-nlp-task-here)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device during training (default: 8)",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per device during evaluation (default: 8)",
    )
    parser.add_argument(
        "--skip_push_to_hub",
        action="store_true",
        default=False,
        help="If set, skip pushing model/tokenizer checkpoints to the Hugging Face Hub (default: False)",
    )
    parser.add_argument(
        "--rare_threshold",
        type=int,
        default=50,
        help="Labels with frequency below this count are classified as 'rare' (default: 50). "
        "Lower this for small/sample datasets so both the rare and not-rare partitions are non-empty.",
    )
    parser.add_argument(
        "--label_space",
        type=str,
        default="label",
        choices=["label", "cluster", "candidate"],
        help="'label' (default) trains against raw labels as usual. 'cluster' trains a router "
        "against the label-cluster space instead (Phase 3 of the full-scale candidate-generator "
        "plan, see README_AMAZON3M.md). 'candidate' trains the 'fine' stage against a bounded "
        "per-document candidate set (true positives + same-cluster hard negatives, fixed size "
        "--candidate_budget) instead of the full label space -- pair with --label_embed_rank at "
        "full scale. Both 'cluster' and 'candidate' require labels_partition.json and "
        "cluster_assignments.json (from scripts/build_label_clusters.py) to already exist under "
        "--base_dir.",
    )
    parser.add_argument(
        "--candidate_budget",
        type=int,
        default=256,
        help="Fixed number of candidate labels (true positives + same-cluster hard negatives, "
        "padded) scored per document when --label_space=candidate (default: 256).",
    )
    parser.add_argument(
        "--label_embed_rank",
        type=int,
        default=None,
        help="If set, factorizes MultilabelICDClassifier/MultilabelICDPlantClassifier's per-label "
        "parameters (label_embeddings, boost_mul, boost_add) into a small (num_labels, rank) base "
        "table plus a shared up-projection, instead of a full (num_labels, hidden_size) table each. "
        "Needed at Amazon-3M's full ~2.81M-label scale (see README_AMAZON3M.md's full-scale "
        "candidate-generator plan); leave unset (dense) for a bounded num_labels like a cluster-router.",
    )
    parser.add_argument(
        "--label_coverage_frac",
        type=float,
        default=0.001,
        help="Fraction of the (train/eval) data initially sampled before greedily topping up to full "
        "label coverage (default: 0.001, tuned for MIMIC-IV's ~270K-row full dataset). On a small "
        "dataset this rounds to ~0 rows, so the greedy top-up alone determines the final size -- "
        "pass 1.0 to use the full train/eval split as-is.",
    )

    return parser.parse_args()


def setup_logging(**kwargs):
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

    # Optionally, configure a specific third-party logger (e.g., 'bertopic')
    # third_party_logger = logging.getLogger('bertopic')
    # third_party_logger.setLevel(logging.DEBUG)  # Capture all messages from this library
    # third_party_logger.addHandler(file_handler)
    # Uncomment the following line if you wish to prevent propagation to the root logger:
    # third_party_logger.propagate = False
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


# Function to Authenticate with Hugging Face
def authenticate_hf(token):
    """Authenticate with Hugging Face using a token."""
    login(token=token)
    print("Authenticated with Hugging Face Hub")


# Function to Get Data from Source-Dir
def get_data(source, data):
    """
    Load CSV data into a pandas DataFrame.

    Constructs the file path using the provided `data` and `source` parameters,
    reads the CSV file, and ensures that the 'text' and 'labels' columns are strings.

    Args:
        source (str): The source directory or identifier used to locate the CSV file.
        data (str): The base filename (without extension) of the CSV data.

    Returns:
        pd.DataFrame: A DataFrame containing the CSV data with 'text' and 'labels' columns.
    """
    # Construct the file path with a '.csv' extension.
    filepath = join_path_file(data, source, ext=".csv")

    # Read the CSV file, selecting only the 'text' and 'labels' columns,
    # and explicitly define their data types as strings.
    df = pd.read_csv(
        filepath,
        header=0,
        usecols=["text", "labels", "is_valid"],
        dtype={"text": str, "labels": str, "is_valid": bool},
    )

    # Convert the selected columns to string type (if needed).
    df[["text", "labels"]] = df[["text", "labels"]].astype(str)

    return df


# Function to define the custom model
def define_model_old(base_model, num_labels):
    """Define the multilabel ICD classification model with label-specific attention."""

    class MultilabelICDClassifier(nn.Module):
        def __init__(self, base_model, num_labels):
            super().__init__()
            self.base_model = base_model
            self.hidden_size = base_model.config.hidden_size  # e.g., 768 for BERT-base
            self.num_labels = num_labels

            # Freeze base model weights
            for param in self.base_model.parameters():
                param.requires_grad = False

            # Label embedding matrix (queries for attention)
            self.label_embeddings = nn.Parameter(
                torch.randn(
                    num_labels, self.hidden_size
                )  # Shape: (num_labels, hidden_size)
            )

            # Multi-Head Attention with label-specific queries
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=8,  # Fixed number of heads (must divide hidden_size)
                batch_first=True,
            )

            # Learnable parameters for boosting
            self.boost_mul = nn.Parameter(
                torch.ones(num_labels, self.hidden_size)  # Multiplicative factor
            )
            self.boost_add = nn.Parameter(
                torch.zeros(num_labels, self.hidden_size)  # Additive factor
            )

            # Pooling layer to reduce sequence dimension
            self.pooling = nn.AdaptiveAvgPool1d(
                1
            )  # Average pooling over sequence length

        def forward(self, input_ids, attention_mask=None, labels=None):
            # Get base model outputs (frozen)
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            # Full sequence hidden states from the last layer
            hidden_states = outputs.hidden_states[
                -1
            ]  # Shape: (batch_size, seq_len, hidden_size)

            # Prepare attention mask for padding tokens
            if attention_mask is not None:
                attn_mask = (
                    attention_mask == 0
                )  # Shape: (batch_size, seq_len), True for padding
            else:
                attn_mask = None

            # Expand label embeddings to match batch size
            batch_size = hidden_states.size(0)
            label_queries = self.label_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # Shape: (batch_size, num_labels, hidden_size)

            # Multi-Head Attention: Label-specific attention
            attn_output, attn_weights = self.attention(
                query=label_queries,  # (batch_size, num_labels, hidden_size)
                key=hidden_states,  # (batch_size, seq_len, hidden_size)
                value=hidden_states,  # (batch_size, seq_len, hidden_size)
                key_padding_mask=attn_mask,  # Mask padding tokens
            )  # attn_output: (batch_size, num_labels, hidden_size)

            # Boost attention output with element-wise multiplication and addition
            # boost_mul_expanded = self.boost_mul.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_labels, hidden_size)
            # boost_add_expanded = self.boost_add.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_labels, hidden_size)
            boosted_attn_output = torch.addcmul(
                self.boost_add,  # Additive term
                attn_output,  # Base tensor
                self.boost_mul,  # Multiplicative term
                value=1.0,  # Scaling factor for multiplication
            )

            # Pool across the hidden_size dimension to get per-label scores
            pooled_output = self.pooling(
                boosted_attn_output
            )  # Shape: (batch_size, num_labels, 1)
            logits = pooled_output.squeeze(-1)  # Shape: (batch_size, num_labels)

            # Compute loss if labels are provided
            loss = None
            if labels is not None:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels.float())

            return (
                {"loss": loss, "logits": logits, "attn_weights": attn_weights}
                if loss is not None
                else {"logits": logits, "attn_weights": attn_weights}
            )

    return MultilabelICDClassifier(base_model, num_labels)


# Function to define the custom model
def define_model(base_model, num_labels, num_layers_to_unfreeze=0, label_embed_rank=None):
    """Define the multilabel ICD classification model with label-specific attention.

    label_embed_rank: None (default) keeps every per-label parameter
    (label_embeddings, boost_mul, boost_add) at its original dense
    (num_labels, hidden_size) size -- safe when num_labels is bounded (e.g.
    a cluster-router). Set this for the full ~2.81M-label Amazon-3M
    vocabulary (see README_AMAZON3M.md's full-scale candidate-generator
    plan) to factorize each into a small (num_labels, label_embed_rank)
    base table plus a shared up-projection, exactly mirroring
    LTRModel.label_embed_rank in xcube/l2r/models/core.py.
    """
    from torch.nn.init import trunc_normal_

    class MultilabelICDClassifier(nn.Module):
        def __init__(self, base_model, num_labels, pooling_output_size=128):
            super().__init__()
            self.base_model = base_model
            self.hidden_size = base_model.config.hidden_size
            self.num_labels = num_labels
            self.pooling_output_size = pooling_output_size  # e.g., 128

            # Modify the config's __dict__ to include your custom attributes
            # This ensures they'll be saved with the model
            self.config = base_model.config
            config_dict = self.config.__dict__

            # Core model configuration
            config_dict["num_labels"] = num_labels
            config_dict["model_type"] = "multilabel_icd_classifier"
            config_dict["_name_or_path"] = base_model.config._name_or_path
            config_dict["label_embed_rank"] = label_embed_rank

            # Custom layer metadata
            config_dict["label_embedding_init"] = (
                "randn"  # Initialization for label_embeddings
            )
            config_dict["attention_num_heads"] = 8  # Multi-head attention heads
            config_dict["attention_batch_first"] = True  # Batch-first attention
            config_dict["boost_mul_init"] = "ones"  # Boost multiplicative init
            config_dict["boost_add_init"] = "zeros"  # Boost additive init
            config_dict["pooling_type"] = "adaptive_avg_pool1d"  # Pooling layer type
            config_dict["pooling_output_size"] = (
                self.pooling_output_size
            )  # Pooling output size
            config_dict["num_layers_to_unfreeze"] = num_layers_to_unfreeze

            # Freeze base model weights
            # logging.info(
            #     f"Creating Multi-Label Classification Model from base model {self.base_model.model.model.name_or_path}"
            # )
            # self._freeze_all()
            # logging.info(f"After freezing all the parameters of the base model")
            logging.info(
                f"Will now start to create Multilabel-Classification Model from the base model"
            )
            compute_trainable_params(self, recursive_depth=0)

            # self._unfreeze_lora()
            # logging.info(f"After unfreezing just the lora parameters of the base model")
            # self.compute_trainable_params()

            # self._unfreeze_layers(self.config.num_layers_to_unfreeze)
            # logging.info(f"After unfreezing the last {self.config.num_layers_to_unfreeze} of the base model")
            # self.compute_trainable_params()

            # Label embedding matrix (queries for attention). Factorized into a
            # small (num_labels, label_embed_rank) base table + one shared
            # up-projection when label_embed_rank is set -- a full-rank
            # (num_labels, hidden_size) table is only safe when num_labels is
            # bounded (e.g. a cluster-router); at Amazon-3M's full ~2.81M-label
            # scale it's tens of GB of parameters plus optimizer state (see
            # README_AMAZON3M.md's full-scale candidate-generator plan).
            self.label_embed_rank = label_embed_rank
            if self.label_embed_rank:
                self.label_embeddings = nn.Embedding(num_labels, self.label_embed_rank)
                self.label_up_proj = nn.Linear(self.label_embed_rank, self.hidden_size, bias=False)
                trunc_normal_(self.label_embeddings.weight, std=0.01)
                trunc_normal_(self.label_up_proj.weight, std=0.01)
            else:
                self.label_embeddings = nn.Embedding(num_labels, self.hidden_size)
                self.label_up_proj = None
                trunc_normal_(self.label_embeddings.weight, std=0.01)

            # Multi-Head Attention with label-specific queries
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=self.config.attention_num_heads,  # Fixed number of heads (must divide hidden_size)
                batch_first=self.config.attention_batch_first,
            )

            # Learnable parameters for boosting -- same factorization as
            # label_embeddings above when label_embed_rank is set. The shared
            # projection's bias recovers the original uniform init (boost_mul
            # ~= 1, boost_add ~= 0 for every label) while the low-rank
            # per-label base term (initialized to zero) is free to diverge
            # during training.
            if self.label_embed_rank:
                self.boost_mul_base = nn.Embedding(num_labels, self.label_embed_rank)
                self.boost_mul_proj = nn.Linear(self.label_embed_rank, self.hidden_size)
                nn.init.zeros_(self.boost_mul_base.weight)
                nn.init.zeros_(self.boost_mul_proj.weight)
                nn.init.ones_(self.boost_mul_proj.bias)

                self.boost_add_base = nn.Embedding(num_labels, self.label_embed_rank)
                self.boost_add_proj = nn.Linear(self.label_embed_rank, self.hidden_size)
                nn.init.zeros_(self.boost_add_base.weight)
                nn.init.zeros_(self.boost_add_proj.weight)
                nn.init.zeros_(self.boost_add_proj.bias)

                self.boost_mul = None
                self.boost_add = None
            else:
                self.boost_mul = nn.Parameter(
                    torch.ones(num_labels, self.hidden_size)  # Multiplicative factor
                )
                self.boost_add = nn.Parameter(
                    torch.zeros(num_labels, self.hidden_size)  # Additive factor
                )
                self.boost_mul_base = self.boost_mul_proj = None
                self.boost_add_base = self.boost_add_proj = None

            # Pooling layer over hidden_size
            self.pooling = nn.AdaptiveAvgPool1d(self.config.pooling_output_size)

            # Linear projection to single logit per label
            self.classifier = nn.Linear(self.pooling_output_size, 1)

            logging.info(
                f"Creating the Multi-Label Classification Model from base model {self.base_model.model.model.name_or_path} completed!!!"
            )
            compute_trainable_params(self, recursive_depth=0)
            # compute_trainable_params(self, recursive_depth=2, finegrained=True)
            # compute_trainable_params(self.base_model)

        def compute_trainable_params(self):
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.parameters())
            trainable_percentage = 100 * trainable_params / total_params

            logging.info(
                f"📊 trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percentage:.4f}"
            )
            return trainable_params, total_params

        def _freeze_all(self):
            """Freeze all base model weights"""
            for param in self.base_model.parameters():
                param.requires_grad = False

        def _unfreeze_lora(self):
            """Unfreeze LoRA parameters in the Mistral model."""
            for name, param in self.base_model.named_parameters():
                if "lora" in name.lower():  # Match LoRA params (e.g., lora_A, lora_B)
                    param.requires_grad = True
            #     lora_config = LoraConfig(
            #     r=16,
            #     lora_alpha=32,
            #     target_modules=["q_proj", "v_proj"],
            #     lora_dropout=0.05,
            #     bias="none",
            #     task_type="CAUSAL_LM"
            # )
            #     self.base_model = get_peft_model(self.base_model, lora_config)
            #     self.base_model.print_trainable_parameters()  # Optional: for verification

        def _unfreeze_layers(self, num_layers_to_unfreeze):
            """Unfreeze the last N layers of Mistral."""
            total_layers = len(self.base_model.model.model.layers)  # 32 for Mistral-7B
            for i, layer in enumerate(self.base_model.model.model.layers):
                if i >= total_layers - num_layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True

        def forward(self, input_ids, attention_mask=None, labels=None, candidate_label_ids=None, candidate_mask=None):
            """
            candidate_label_ids (optional): per-example label ids to score against,
                shape (batch_size, budget), instead of every one of self.num_labels
                labels -- the mechanism that keeps training tractable at Amazon-3M's
                full ~2.81M-label scale (see README_AMAZON3M.md's full-scale
                candidate-generator plan). Omit to keep scoring every label, as
                before (e.g. for a bounded cluster-router num_labels).
            candidate_mask (optional): shape (batch_size, budget), True where
                candidate_label_ids is a real candidate and False where it's
                padding. Only meaningful together with candidate_label_ids.
            """
            # Get base model outputs
            with (
                torch.no_grad()
                if not any(p.requires_grad for p in self.base_model.parameters())
                else torch.enable_grad()
            ):
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            # Full sequence hidden states from the last layer
            hidden_states = outputs.hidden_states[
                -1
            ]  # Shape: (batch_size, seq_len, hidden_size)

            # Prepare attention mask for padding tokens
            if attention_mask is not None:
                attn_mask = (
                    attention_mask == 0
                )  # Shape: (batch_size, seq_len), True for padding
            else:
                attn_mask = None

            # Score every label by default; a per-example candidate_label_ids
            # restricts scoring to just those ids instead (see the docstring above).
            batch_size = hidden_states.size(0)
            if candidate_label_ids is not None:
                label_ids = candidate_label_ids  # (batch_size, budget)
            else:
                label_ids = (
                    torch.arange(self.num_labels, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )  # (batch_size, num_labels)

            label_base = self.label_embeddings(label_ids)  # (batch_size, n, label_embed_rank or hidden_size)
            label_queries = (
                self.label_up_proj(label_base) if self.label_up_proj is not None else label_base
            )  # (batch_size, n, hidden_size)

            # Multi-Head Attention: Label-specific attention
            attn_output, attn_weights = self.attention(
                query=label_queries,  # (batch_size, n, hidden_size)
                key=hidden_states,  # (batch_size, seq_len, hidden_size)
                value=hidden_states,  # (batch_size, seq_len, hidden_size)
                key_padding_mask=attn_mask,  # Mask padding tokens
            )  # attn_output: (batch_size, n, hidden_size)

            # Boost attention output with element-wise multiplication and addition.
            # In factorized mode (label_embed_rank set), boost_mul/boost_add are
            # gathered per label_ids just like label_queries above -- a
            # (batch_size, n, hidden_size) tensor instead of a single
            # (num_labels, hidden_size) parameter shared/broadcast across the batch.
            if self.label_embed_rank:
                boost_mul = self.boost_mul_proj(self.boost_mul_base(label_ids))
                boost_add = self.boost_add_proj(self.boost_add_base(label_ids))
            else:
                boost_mul = self.boost_mul  # (num_labels, hidden_size); broadcasts against attn_output
                boost_add = self.boost_add
            boosted_attn_output = torch.addcmul(
                boost_add,  # Additive term
                attn_output,  # Base tensor
                boost_mul,  # Multiplicative term
                value=1.0,  # Scaling factor for multiplication
            )

            # (Method 1) Pool across the hidden_size dimension to get per-label scores
            pooled_output = self.pooling(
                boosted_attn_output
            )  # Shape: (batch_size, n, pooling_output_size)
            # Project to logits
            logits = self.classifier(pooled_output).squeeze(
                -1
            )  # Shape: (batch_size, n, pooling_output_size) -> (batch_size, n, 1) -> (batch_size, n)

            # (Method 2) Sum pooling over hidden_size dimension to get per-label scores
            # logits = boosted_attn_output.sum(dim=-1)  # (batch_size, num_labels)

            # Compute loss if labels are provided
            loss = None
            if labels is not None:
                loss_fn = nn.BCEWithLogitsLoss()
                # loss = loss_fn(logits, labels.float())
                #####
                batch_preds, batch_stds = logits, labels.float()
                if candidate_mask is not None:
                    # Here each row is one document and columns are its own candidate
                    # labels (true positives + same-cluster hard negatives), unlike
                    # LTRModel's masked_approxNDCG_loss where rows are (example,
                    # candidate-slot) pairs -- so padding here means some columns
                    # within a row are fake, not that whole rows are. Push padding
                    # columns to the bottom of the rank via a very low predicted
                    # score and zero relevance, rather than excluding rows.
                    batch_preds = batch_preds.masked_fill(~candidate_mask, -1e4)
                    batch_stds = batch_stds * candidate_mask.float()
                loss = fixed_approxNDCG_loss(
                    batch_preds=batch_preds,  # Model output logits (batch_size, n)
                    batch_stds=batch_stds,  # One-hot encoded labels (batch_size, n)
                    label_type=LABEL_TYPE.MultiLabel,  # Multilabel setting
                    gpu=torch.cuda.is_available(),  # Use GPU if available
                    alpha=10,  # Default alpha, adjust if needed
                )
                #####

            return (
                {"loss": loss, "logits": logits, "attn_weights": attn_weights}
                if loss is not None
                else {"logits": logits, "attn_weights": attn_weights}
            )

    class MultilabelICDPlantClassifier(nn.Module):
        def __init__(self, base_model, num_labels, pooling_output_size=128):
            super().__init__()
            self.base_model = base_model
            self.hidden_size = base_model.config.hidden_size
            self.num_labels = num_labels
            self.pooling_output_size = pooling_output_size  # e.g., 128

            # Modify the config's __dict__ to include custom attributes
            self.config = base_model.config
            config_dict = self.config.__dict__

            # Core model configuration
            config_dict["num_labels"] = num_labels
            config_dict["top_model_type"] = "multilabel_icd_plant_classifier"
            config_dict["_name_or_path"] = base_model.config._name_or_path
            config_dict["label_embed_rank"] = label_embed_rank

            # Custom layer metadata
            config_dict["boost_mul_init"] = "ones"  # Boost multiplicative init
            config_dict["boost_add_init"] = "zeros"  # Boost additive init
            config_dict["pooling_type"] = "adaptive_avg_pool1d"  # Pooling layer type
            config_dict["pooling_output_size"] = self.pooling_output_size
            config_dict["num_layers_to_unfreeze"] = num_layers_to_unfreeze

            logging.info(
                f"Starting creation of Multilabel-Classification Model from base model"
            )
            compute_trainable_params(self, recursive_depth=0)

            # Learnable parameters for boosting. Factorized into a small
            # (num_labels, label_embed_rank) base table + shared projection when
            # label_embed_rank is set, exactly mirroring MultilabelICDClassifier
            # and LTRModel.label_embed_rank (xcube/l2r/models/core.py) -- a
            # full-rank (num_labels, hidden_size) parameter is only safe when
            # num_labels is bounded (e.g. a cluster-router); see
            # README_AMAZON3M.md's full-scale candidate-generator plan. The
            # shared projection's bias recovers the original uniform init
            # (boost_mul ~= 1, boost_add ~= 0) while the low-rank per-label base
            # term (initialized to zero) is free to diverge during training.
            self.label_embed_rank = label_embed_rank
            if self.label_embed_rank:
                self.boost_mul_base = nn.Embedding(num_labels, self.label_embed_rank)
                self.boost_mul_proj = nn.Linear(self.label_embed_rank, self.hidden_size)
                nn.init.zeros_(self.boost_mul_base.weight)
                nn.init.zeros_(self.boost_mul_proj.weight)
                nn.init.ones_(self.boost_mul_proj.bias)

                self.boost_add_base = nn.Embedding(num_labels, self.label_embed_rank)
                self.boost_add_proj = nn.Linear(self.label_embed_rank, self.hidden_size)
                nn.init.zeros_(self.boost_add_base.weight)
                nn.init.zeros_(self.boost_add_proj.weight)
                nn.init.zeros_(self.boost_add_proj.bias)

                self.boost_mul = None
                self.boost_add = None
            else:
                self.boost_mul = nn.Parameter(
                    torch.ones(num_labels, self.hidden_size)  # Multiplicative factor
                )
                self.boost_add = nn.Parameter(
                    torch.zeros(num_labels, self.hidden_size)  # Additive factor
                )
                self.boost_mul_base = self.boost_mul_proj = None
                self.boost_add_base = self.boost_add_proj = None

            # Pooling layer over hidden_size
            self.pooling = nn.AdaptiveAvgPool1d(self.config.pooling_output_size)

            # Linear projection to single logit per label
            self.classifier = nn.Linear(self.pooling_output_size, 1)

            logging.info(
                f"Completed creation of Multi-Label Classification Model from base model {self.base_model.config._name_or_path}!!!"
            )
            compute_trainable_params(self, recursive_depth=0)
            # my_model = self.base_model.ground_model.base_model.model
            # compute_trainable_params(self, recursive_depth=1, finegrained=True)
            # compute_trainable_params(self.base_model)

        def compute_trainable_params(self):
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.parameters())
            trainable_percentage = 100 * trainable_params / total_params

            logging.info(
                f"📊 trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percentage:.4f}"
            )
            return trainable_params, total_params

        def forward(self, input_ids, attention_mask=None, labels=None, candidate_label_ids=None, candidate_mask=None):
            """
            candidate_label_ids (optional): the SAME per-example label ids passed
                to the upstream LTRModel base_model call that produced
                outputs["logits"] -- needed here only to gather the matching rows
                of the factorized boost_mul/boost_add tables (when
                label_embed_rank is set), since attn_weights' last axis is
                whatever candidate/full-label axis the base model produced, not
                necessarily 0..num_labels-1. See README_AMAZON3M.md's full-scale
                candidate-generator plan.
            candidate_mask (optional): shape (batch_size, budget), True where
                candidate_label_ids is a real candidate and False where it's
                padding. Only meaningful together with candidate_label_ids.
            """
            # Get base model outputs. candidate_label_ids/candidate_mask must be
            # forwarded here too -- without them the upstream LTRModel falls
            # back to scoring every one of its own num_labels, while the
            # boosting step below gathers boost_mul/boost_add at the (possibly
            # much narrower) candidate width, causing a shape mismatch in
            # torch.addcmul. Unlike MultilabelICDClassifier (which does its own
            # label-attention internally against a label-count-agnostic base
            # model), this classifier's base_model IS an LTR model that does
            # its own per-label scoring, so the candidate restriction has to
            # propagate through it too.
            with (
                torch.no_grad()
                if not any(p.requires_grad for p in self.base_model.parameters())
                else torch.enable_grad()
            ):
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    candidate_label_ids=candidate_label_ids,
                    candidate_mask=candidate_mask,
                )

            # Extract hidden states and attention weights
            hidden_states = outputs["hidden_states"]  # Shape: (batch_size, max_len, hidden_size)
            attn_weights = outputs["logits"]  # Shape: (batch_size, max_len, n)

            # Normalize attention weights over max_len dimension
            attn_weights = torch.softmax(attn_weights, dim=1)  # Shape: (batch_size, max_len, n)

            # Compute attention-weighted output as linear combination
            # Reshape attn_weights to (batch_size, n, max_len) for matrix multiplication
            attn_weights = attn_weights.transpose(1, 2)  # Shape: (batch_size, n, max_len)
            attn_output = torch.bmm(
                attn_weights, hidden_states
            )  # Shape: (batch_size, n, hidden_size)

            # Apply boosting. In factorized mode (label_embed_rank set),
            # boost_mul/boost_add are gathered per label id, same as
            # MultilabelICDClassifier -- candidate_label_ids when given (must
            # match whatever candidate set the upstream base_model call used),
            # else every label in order.
            if self.label_embed_rank:
                batch_size = hidden_states.size(0)
                if candidate_label_ids is not None:
                    label_ids = candidate_label_ids  # (batch_size, budget)
                else:
                    label_ids = (
                        torch.arange(self.num_labels, device=hidden_states.device)
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )  # (batch_size, num_labels)
                boost_mul = self.boost_mul_proj(self.boost_mul_base(label_ids))
                boost_add = self.boost_add_proj(self.boost_add_base(label_ids))
            else:
                boost_mul = self.boost_mul  # (num_labels, hidden_size); broadcasts against attn_output
                boost_add = self.boost_add
            boosted_attn_output = torch.addcmul(
                boost_add,  # Additive term
                attn_output,  # Base tensor
                boost_mul,  # Multiplicative term
                value=1.0,  # Scaling factor
            )

            # Pool across hidden_size dimension to get per-label scores
            pooled_output = self.pooling(
                boosted_attn_output
            )  # Shape: (batch_size, n, pooling_output_size)

            # Project to logits
            logits = self.classifier(pooled_output).squeeze(
                -1
            )  # Shape: (batch_size, n, pooling_output_size) -> (batch_size, n, 1) -> (batch_size, n)

            # Compute loss if labels provided
            loss = None
            if labels is not None:
                loss_fn = nn.BCEWithLogitsLoss()
                # loss = loss_fn(logits, labels.float())
                #####
                batch_preds, batch_stds = logits, labels.float()
                if candidate_mask is not None:
                    # Same column-masking approach as MultilabelICDClassifier --
                    # each row is one document, columns are its own candidates, so
                    # padding means some columns within a row are fake.
                    batch_preds = batch_preds.masked_fill(~candidate_mask, -1e4)
                    batch_stds = batch_stds * candidate_mask.float()
                loss = fixed_approxNDCG_loss(
                    batch_preds=batch_preds,  # Model output logits (batch_size, n)
                    batch_stds=batch_stds,  # One-hot encoded labels (batch_size, n)
                    label_type=LABEL_TYPE.MultiLabel,  # Multilabel setting
                    gpu=torch.cuda.is_available(),  # Use GPU if available
                    alpha=10,  # Default alpha, adjust if needed
                )
                #####

            return (
                {"loss": loss, "logits": logits, "attn_weights": attn_weights}
                if loss is not None
                else {"logits": logits, "attn_weights": attn_weights}
            )

    # Model selection based on base_model.config.model_type
    if hasattr(base_model.config, 'model_type') and base_model.config.model_type == 'ltr_model':
        logging.info("🌱🏥 Creating MultilabelICDPlantClassifier - Specialized for planted attention classification of medical codes! 🌿✨")
        return MultilabelICDPlantClassifier(base_model, num_labels)
    else:
        logging.info("🏥📊 Creating MultilabelICDClassifier - Standard multilabel medical classifier! 🔬💫")
        return MultilabelICDClassifier(base_model, num_labels)

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


def chunk_text(
    text: str,
    is_valid: bool,
    labels: list,  # One-hot encoded list of length ~7942, or a fixed-size candidate-budget target in candidate mode
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
        is_valid (bool): Whether the text is in training or validation set.
        labels (list): One-hot encoded list of labels (0s and 1s).
        tokenizer (AutoTokenizer): Tokenizer to encode text.
        max_length (int): Maximum tokens per chunk (including special tokens).
        overlap (int): Number of tokens to overlap between chunks.

    Returns:
        List[Dict]: List of tokenized chunks with input_ids, attention_mask, etc.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    chunk_size = max_length - 2  # For [CLS] and [SEP]
    if chunk_size <= 0:
        raise ValueError("max_length must be at least 3")

    chunks = []
    i = 0
    while i < len(input_ids):
        end = min(i + chunk_size, len(input_ids))
        chunk_ids = input_ids[i:end]
        chunk_mask = attention_mask[i:end]
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

        # Pad to max_length
        padding_length = max_length - len(chunk_ids)
        if padding_length > 0:
            chunk_ids = torch.cat(
                [chunk_ids, torch.zeros(padding_length, dtype=chunk_ids.dtype)]
            )
            chunk_mask = torch.cat(
                [chunk_mask, torch.zeros(padding_length, dtype=chunk_mask.dtype)]
            )

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
            # np.asarray(..., dtype=np.uint8), not the bare python list `labels`
            # is: this dict is later collected into flattened_chunks and built
            # into the final Dataset via Dataset.from_list (see the "Flattening
            # chunks" step below preprocess_data). By the time `labels` reaches
            # here it has already round-tripped through dataset.map()'s default
            # (non-torch) decoding of the Dataset.from_dict'd dataset above --
            # which returns plain Python ints regardless of that dataset's own
            # storage dtype, so the earlier uint8 fix there doesn't survive to
            # this point. A bare python int list gets pyarrow-inferred as
            # list<item: int64> at Dataset.from_list, and at Wiki10-31K full
            # scale that's large enough on its own to hit the same ~2GB
            # per-buffer offset-overflow this file's other porting note
            # documents -- worse, every long document's chunks each carry a
            # full copy of that document's num_labels-wide vector, so the
            # effective row count feeding this buffer is total chunks, not
            # total docs. numpy arrays get the same dtype-preserving treatment
            # pyarrow gives torch tensors (both go through datasets'
            # cast_to_python_objects), so this keeps the child buffer ~8x
            # smaller through this second, later construction point too.
            "labels": np.asarray(labels, dtype=np.uint8),
        }
        # Only add these keys in candidate mode. If every chunk dict always
        # carried them (even as None in label/cluster mode), Dataset.from_list
        # would infer a schema that includes candidate_label_ids/candidate_mask
        # columns regardless of label_space -- and CustomTrainer.evaluate()'s
        # `"candidate_label_ids" in eval_dataset.column_names` mode-detection
        # would then misfire True in label/cluster mode too, crashing with
        # KeyError on a batch that (correctly) never got those columns from
        # set_format (see README_AMAZON3M.md, Phase 7).
        if candidate_label_ids is not None:
            chunk_dict["candidate_label_ids"] = candidate_label_ids
            chunk_dict["candidate_mask"] = candidate_mask
        chunks.append(chunk_dict)
        i += chunk_size - overlap

    return chunks


def load_cluster_assignments(base_dir):
    "Load the flat label clustering built by scripts/build_label_clusters.py."
    with open(os.path.join(base_dir, "cluster_assignments.json"), "r") as f:
        cluster_assignments = json.load(f)
    return cluster_assignments["cluster_id"], cluster_assignments["num_clusters"]


def build_label_or_cluster_vectors(labels, base_dir, rare_threshold, label_space, is_valid=None):
    """Encode `labels` (a list of per-doc label-string lists) into a sparse
    multi-hot matrix, in one of two spaces:

    - "label" (default): fits a fresh MultiLabelBinarizer over the raw label
      strings and (re)writes labels_partition.json -- the original behavior.
    - "cluster": trains a router instead (Phase 3 of the full-scale
      candidate-generator plan, see README_AMAZON3M.md). Reuses the label
      vocabulary + flat clustering already fixed by a prior "label"-space
      Stage 2 run plus scripts/build_label_clusters.py (both must already
      exist under `base_dir`), maps each doc's true labels to the set of
      clusters they fall in, and binarizes over cluster ids instead of raw
      labels. Writes cluster_labels_partition.json, never overwriting the
      label-space labels_partition.json those artifacts were built from.

    Returns (mlb, label_vectors, rare_labels, not_rare_labels, zero_labels).

    Label buckets are keyed off TRAINING-only frequency when `is_valid` (a
    per-row bool list, True == eval/held-out) is supplied, so that "zero-shot"
    (`zero_labels`) means exactly zero *training* occurrences even though the
    code appears in the eval split. Without `is_valid` it falls back to the
    old total-frequency behavior. Buckets: zero (train freq == 0) subset of
    rare (train freq < rare_threshold); not_rare/head = train freq >= threshold.
    """
    if label_space == "label":
        # sparse_output=True avoids ever materializing a dense
        # (num_docs, num_labels) array -- at Amazon-3M's full scale (2.46M
        # docs x 2.81M labels) the dense form is tens of terabytes.
        mlb = MultiLabelBinarizer(sparse_output=True)
        label_vectors = mlb.fit_transform(labels)  # scipy.sparse.csr_matrix
        partition_file = os.path.join(base_dir, "labels_partition.json")
    elif label_space == "cluster":
        labels_file = os.path.join(base_dir, "labels_partition.json")
        with open(labels_file, "r") as f:
            mlb_classes = json.load(f)["classes"]
        cluster_ids, num_clusters = load_cluster_assignments(base_dir)
        assert len(cluster_ids) == len(mlb_classes), (
            f"cluster_assignments.json has {len(cluster_ids)} entries, expected "
            f"{len(mlb_classes)} (one per label in {labels_file}) -- was "
            f"build_label_clusters.py run against a stale labels_partition.json?"
        )
        label_to_cluster = dict(zip(mlb_classes, cluster_ids))
        cluster_names = [str(i) for i in range(num_clusters)]
        cluster_label_sets = [
            sorted(set(str(label_to_cluster[l]) for l in doc_labels if l in label_to_cluster))
            for doc_labels in labels
        ]
        mlb = MultiLabelBinarizer(classes=cluster_names, sparse_output=True)
        mlb.fit([])  # classes are fixed by the clustering; no data needed to determine them
        label_vectors = mlb.transform(cluster_label_sets)
        partition_file = os.path.join(base_dir, "cluster_labels_partition.json")
    else:
        raise ValueError(f"Unknown label_space {label_space!r}, expected 'label' or 'cluster'")

    # Calculate label frequencies without densifying: sparse .sum(axis=0)
    # returns a (1, num_labels) numpy.matrix.
    # Total (train+eval) frequency, kept for reference.
    label_frequencies = np.asarray(label_vectors.sum(axis=0)).ravel()
    # Bucketing frequency: TRAIN-only when is_valid is provided (True == eval),
    # so zero-shot = 0 *training* examples. Falls back to total otherwise.
    if is_valid is not None:
        train_mask = np.array([not bool(v) for v in is_valid])
        bucket_freq = np.asarray(label_vectors[train_mask].sum(axis=0)).ravel()
    else:
        bucket_freq = label_frequencies

    zero_labels = [i for i, freq in enumerate(bucket_freq) if freq == 0]
    rare_labels = [
        i for i, freq in enumerate(bucket_freq) if freq < rare_threshold
    ]
    not_rare_labels = [
        i for i, freq in enumerate(bucket_freq) if freq >= rare_threshold
    ]
    num_labels = len(mlb.classes_)
    assert num_labels == (
        len(rare_labels) + len(not_rare_labels)
    ), f"Expected rare+not_rare to be {num_labels} labels, got {(len(rare_labels) + len(not_rare_labels))}"

    # Create dictionaries with label names and (train) frequencies
    rare_labels_dict = {
        mlb.classes_[i]: int(bucket_freq[i]) for i in rare_labels
    }
    not_rare_labels_dict = {
        mlb.classes_[i]: int(bucket_freq[i]) for i in not_rare_labels
    }
    zero_labels_dict = {
        mlb.classes_[i]: int(bucket_freq[i]) for i in zero_labels
    }

    # Combine into a single dictionary. "zero" is a subset of "rare"; "head" is
    # exactly "not_rare"; "few" = rare minus zero (derived by consumers).
    labels_partition = {
        "rare": rare_labels_dict,
        "not_rare": not_rare_labels_dict,
        "zero": zero_labels_dict,
        "classes": list(mlb.classes_),  # Save mlb.classes_ as a list
    }

    # Save to a single JSON file
    with open(partition_file, "w") as f:
        json.dump(labels_partition, f, indent=4)

    # Log basic info
    _freq_kind = "train" if is_valid is not None else "total"
    logging.info(f"[{label_space}] Total number of labels: {len(mlb.classes_)}")
    logging.info(f"[{label_space}] Zero-shot labels ({_freq_kind} freq == 0): {len(zero_labels)}")
    logging.info(f"[{label_space}] Rare labels ({_freq_kind} freq < {rare_threshold}, incl. zero): {len(rare_labels)}")
    logging.info(f"[{label_space}] Not rare / head labels ({_freq_kind} freq >= {rare_threshold}): {len(not_rare_labels)}")
    logging.info(f"[{label_space}] Label partitions and classes saved to {partition_file}")

    return mlb, label_vectors, rare_labels, not_rare_labels, zero_labels


def build_candidate_arrays(label_vectors, mlb_classes, base_dir, budget=256, seed=42):
    """Build a fixed-size, per-document candidate set for the full-scale
    candidate-generator plan (see README_AMAZON3M.md): true positive label
    indices plus same-cluster hard negatives (labels sharing a cluster with
    at least one of the doc's true positives), padded/truncated to `budget`.
    This is what lets Stage 2/4/5's "fine" training score a bounded set of
    labels per document instead of every one of Amazon-3M's ~2.81M labels.

    `label_vectors` (sparse, shape (num_docs, num_labels), full label space --
    NOT cluster space) and `mlb_classes` should come from
    `build_label_or_cluster_vectors(..., label_space="label")`, so index i
    here means the same label everywhere (labels_partition.json, the
    clustering, and this). Requires cluster_assignments.json /
    cluster_to_labels.json (scripts/build_label_clusters.py) to already
    exist under `base_dir`.

    Returns (candidate_label_ids, candidate_targets, candidate_mask), each a
    numpy array of shape (num_docs, budget): candidate_label_ids holds the
    actual label index for each slot (0 for pure padding -- a real label id,
    but excluded from anything downstream via candidate_mask being False
    there); candidate_targets is 1.0 at true-positive slots and 0.0
    elsewhere (negatives and padding); candidate_mask is True for every real
    (positive or negative) slot and False only for pure padding.

    NOTE: this is a plain per-row Python loop -- fine for the sample, but
    the real full-corpus (2.46M rows) run would want this vectorized/chunked
    for performance; not a correctness concern, just an unoptimized path for
    now (consistent with other full-scale performance caveats already
    flagged in README_AMAZON3M.md, e.g. per-cluster Stage 3 re-tokenization).
    """
    cluster_ids, num_clusters = load_cluster_assignments(base_dir)
    assert len(cluster_ids) == len(mlb_classes), (
        f"cluster_assignments.json has {len(cluster_ids)} entries, expected "
        f"{len(mlb_classes)} -- was build_label_clusters.py run against a "
        f"stale labels_partition.json?"
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


def remap_candidate_eval_arrays(logits, labels, candidate_label_ids, candidate_mask, num_labels):
    """Scatter candidate-mode's per-row-relative (n_samples, budget) logits/labels
    back into proper (n_samples, num_labels) arrays, so the existing
    compute_metrics_basic/compute_metrics_with_labels (which assume column k
    means the same label in every row) stay meaningful.

    In "candidate" mode, column k of a row's logits/labels means
    candidate_label_ids[row, k] -- a DIFFERENT real label per row -- not a
    fixed global label index the way it does in "label"/"cluster" mode.
    Macro-averaging metrics like sklearn's f1_score(..., average="macro")
    directly over such arrays silently aggregates unrelated labels across
    rows into the same "column", producing numbers that look like real
    (and often badly wrong-looking) scores but aren't measuring anything
    coherent. This undoes that: every real (positive or negative) candidate
    slot gets scattered into its true global column; every label NOT in a
    row's candidate set is scored as confidently absent (a very low logit,
    0 target) -- the honest reflection of candidate-restricted scoring,
    since a two-stage system never predicts positive on a label it never
    shortlisted for that document.
    """
    n_samples = logits.shape[0]
    full_logits = np.full((n_samples, num_labels), -1e4, dtype=np.float32)
    full_labels = np.zeros((n_samples, num_labels), dtype=np.float32)
    for row in range(n_samples):
        real = candidate_mask[row].astype(bool)
        cols = candidate_label_ids[row][real]
        full_logits[row, cols] = logits[row][real]
        full_labels[row, cols] = labels[row][real]
    return full_logits, full_labels


def get_inv_propensity(labels, a=0.55, b=1.5):
    """Per-label inverse propensity scores (Jain et al. 2016's extreme
    multi-label propensity model), reimplemented in plain numpy against this
    repo's own dense eval arrays -- see
    ~/AttentionXML/deepxml/evaluation.py::get_inv_propensity for the
    reference formula. Fit directly from `labels`' own positive-label
    frequency (the eval split) rather than a separately-plumbed train split:
    on this pipeline's sample-sized smoke tests the two are close enough,
    and PSP@k only cares about relative label rarity, not an exact frequency.

    `labels`: dense (n_samples, num_labels) binary array.
    """
    n = labels.shape[0]
    number = labels.sum(axis=0).astype(np.float64)
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)


def get_psp_at_k(probs, labels, inv_w, k):
    """Propensity-scored precision@k: like precision@k, but a hit on a rare
    (high inv_w) label counts for more, and the score is normalized against
    the best any model could do given each row's own true labels -- not a
    flat k. Needed because on a power-law label distribution, vanilla P@k is
    dominated by head labels and says nothing about tail-label recall.

    `probs`/`labels`: dense (n_samples, num_labels) arrays (post any
    candidate-mode remap -- see remap_candidate_eval_arrays).
    """
    n_samples = labels.shape[0]
    top_k_indices = np.argsort(probs, axis=1)[:, -k:]  # Shape: (n_samples, k)
    pred_mask = np.zeros_like(labels, dtype=np.float64)
    rows = np.arange(n_samples)[:, None]
    pred_mask[rows, top_k_indices] = 1.0
    num = (pred_mask * labels * inv_w[None, :]).sum()

    # Ideal denominator: the k highest-propensity labels among each row's OWN
    # true labels (mirrors torch_dcg_at_k-style normalization) -- rows with
    # fewer than k true labels just contribute all of them (padding zeros add
    # nothing to the sum).
    weighted_true = labels * inv_w[None, :]
    ideal_top_k = -np.sort(-weighted_true, axis=1)[:, :k]
    den = ideal_top_k.sum()
    return num / den if den > 0 else 0.0


# Function to preprocess data
def preprocess_data(df, tokenizer, base_dir, max_length=512, rare_threshold=50, label_space="label", candidate_budget=256):
    """Preprocess text and labels into a Dataset object with custom train/test split.

    `label_space="cluster"` trains against the label-cluster router space
    instead of raw labels (Phase 3 of the full-scale candidate-generator
    plan) -- see `build_label_or_cluster_vectors`. `label_space="candidate"`
    trains the "fine" stage against a bounded per-document candidate set
    (true positives + same-cluster hard negatives, fixed-size
    `candidate_budget`) instead of the full label space -- see
    `build_candidate_arrays`. Both require a prior "label"-space Stage 2 run
    plus scripts/build_label_clusters.py to already exist under `base_dir`.
    """
    # Extract texts, labels, and is_valid from the DataFrame
    num = len(df)  # Will remove later, added now for quick iteration
    texts = df["text"].tolist()[:num]
    labels = df["labels"].apply(lambda x: x.split(";") if x else []).tolist()[:num]
    is_valid = df["is_valid"].tolist()[:num]  # Get the split column

    # "candidate" mode still needs the TRUE full label vocabulary (not a
    # budget-sized or cluster-sized one) so num_labels correctly sizes the
    # model's factorized embedding tables -- the per-doc candidate
    # restriction only bounds what each row *scores*, not how many labels
    # exist. Layer build_candidate_arrays on top of the ordinary "label" path
    # rather than introducing a third branch inside build_label_or_cluster_vectors.
    mlb, label_vectors, rare_labels, not_rare_labels, zero_labels = build_label_or_cluster_vectors(
        labels, base_dir, rare_threshold, "label" if label_space == "candidate" else label_space,
        is_valid=is_valid,
    )
    num_labels = len(mlb.classes_)

    if label_space == "candidate":
        candidate_label_ids, candidate_targets, candidate_mask = build_candidate_arrays(
            label_vectors, mlb.classes_, base_dir, budget=candidate_budget
        )
        logging.info(
            f"[candidate] Built {candidate_label_ids.shape[0]} per-doc candidate sets "
            f"(budget={candidate_budget}); avg real (positive+negative) slots per doc: "
            f"{candidate_mask.sum(axis=1).mean():.1f}"
        )
        dataset_dict = {
            "text": texts,
            "labels": [torch.tensor(row, dtype=torch.float32) for row in candidate_targets],
            "candidate_label_ids": [torch.tensor(row, dtype=torch.long) for row in candidate_label_ids],
            "candidate_mask": [torch.tensor(row, dtype=torch.bool) for row in candidate_mask],
            "is_valid": is_valid,
        }
    else:
        # NOTE: this still densifies one row (length num_labels) at a time. That's
        # fine when num_labels is bounded (the sample corpus, or a cluster/router
        # label space per the full-scale candidate-generator plan) but is NOT the
        # fix for training against the full ~2.81M-label Amazon-3M vocabulary --
        # that's exactly what label_space="candidate" (above) is for.
        #
        # dtype=torch.uint8, not float32: labels are 0/1, and Dataset.from_dict
        # below serializes this list-of-tensors column through pyarrow as a
        # list<item: T> array. At Wiki10-31K full scale (20,762 rows x 30,938
        # labels), a float64 (Arrow's inferred width for python floats out of
        # torch.float32 tensors) child buffer is ~5.1GB, over pyarrow's ~2GB
        # per-buffer limit for standard (non-large_list) list arrays -- crashes
        # with "offset overflow while concatenating arrays" at Dataset.from_dict,
        # not at training time. uint8 keeps the same buffer under ~650MB (matches
        # this file's own back-of-envelope "~640MB, trivial" estimate, which
        # implicitly assumed a boolean/byte representation, not the float64 this
        # code actually produced). Every consumer already does labels.float()
        # before use (see the three MultilabelICDClassifier.forward methods
        # above), so this is a storage-only change -- no precision lost since
        # labels are binary, no downstream call site needs to change. Eurlex-4K
        # never hit this (19,314 x 3,956 x 8 bytes = ~611MB, already under the
        # limit even at float64), which is why the smoke test never surfaced it.
        dataset_dict = {
            "text": texts,
            "labels": [
                torch.tensor(label_vec.toarray().ravel(), dtype=torch.uint8)
                for label_vec in label_vectors
            ],
            "is_valid": is_valid,
        }
    dataset = Dataset.from_dict(dataset_dict)

    # Define a tokenization function
    def tokenize_function(examples):
        encodings = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # Include labels and is_valid in the output
        encodings["labels"] = examples["labels"]
        encodings["is_valid"] = examples["is_valid"]
        return encodings

    expected_labels_len = candidate_budget if label_space == "candidate" else num_labels

    def process_batch(examples):
        all_chunks = []
        candidate_cols = zip(examples["candidate_label_ids"], examples["candidate_mask"]) if label_space == "candidate" else itertools.repeat((None, None))
        for (text, is_valid, labels), (cand_ids, cand_mask) in zip(
            zip(examples["text"], examples["is_valid"], examples["labels"]), candidate_cols
        ):
            # Verify labels length
            assert (
                len(labels) == expected_labels_len
            ), f"Expected {expected_labels_len} labels, got {len(labels)}"
            chunks = chunk_text(
                text, is_valid, labels, tokenizer, max_length, overlap=10,
                candidate_label_ids=cand_ids, candidate_mask=cand_mask,
            )

            # Check that every chunk has the same is_valid and labels
            is_valids = [chunk["is_valid"] for chunk in chunks]
            assert all(
                x == is_valids[0] for x in is_valids
            ), "Inconsistent is_valid across chunks"
            the_labels = [chunk["labels"] for chunk in chunks]
            # np.array_equal, not ==: chunk["labels"] is now a numpy array
            # (see the dtype porting note in chunk_text above), and `array ==
            # array` returns an elementwise bool array, not a single bool --
            # bool()/all() on that raises "truth value of an array ... is
            # ambiguous" instead of doing the intended equality check.
            assert all(
                np.array_equal(x, the_labels[0]) for x in the_labels
            ), "Inconsistent labels across chunks"

            all_chunks.append(chunks)
        return {"chunks": all_chunks}

    # Tokenize with a progress bar using map
    tokenized_dataset = dataset.map(
        process_batch, batched=True, desc="Tokenizing texts and labels"
    )

    # Flatten chunks
    flattened_chunks = [
        chunk
        for data in tqdm(tokenized_dataset, desc="Flattening chunks")
        for chunk in data["chunks"]
    ]

    # Correction to the uint8 porting notes above: neither of those fixes
    # was actually enough. The real ceiling is pyarrow's ~2.15B-element
    # (2^31, int32) offsets-array limit for a standard list<T> column --
    # an ELEMENT-COUNT ceiling, not a byte-size one, so narrowing the item
    # dtype alone (float32 -> uint8) can never fix it once the row count is
    # high enough. And the row count here is chunk count, not doc count:
    # Wiki10-31K's articles routinely need multiple max_length=512 windows,
    # so 20,762 docs -> ~130K flattened chunks empirically (a ~6.2x
    # multiplier), each carrying a full expected_labels_len-wide vector --
    # ~130,000 x 30,938 =~ 4.02B elements, well over the 2.15B ceiling
    # regardless of item width. Confirmed directly: Dataset.from_list on
    # ~130K uint8-item rows still hit "Value 2147499394 too large to fit
    # in C integer type" -- 2147499394 is a hair over 2^31-1.
    #
    # Fix: pull "labels" out of every chunk dict and build it as a
    # pyarrow FixedSizeListArray directly, bypassing Dataset.from_list's
    # type inference for this one column. Every row is the same width
    # (expected_labels_len), so a fixed-size list needs no per-row offsets
    # buffer at all -- structurally immune to this failure mode regardless
    # of row count, not just narrower. Confirmed via isolated repro at the
    # real ~130K-row scale: correct values, and ~2s total vs. the ~2GB-out
    # -of-4GB variable-length list<T> approach failing outright.
    labels_flat = np.concatenate([chunk.pop("labels") for chunk in flattened_chunks])
    tokenized_dataset = Dataset.from_list(flattened_chunks)
    labels_array = pa.FixedSizeListArray.from_arrays(
        pa.array(labels_flat, type=pa.uint8()), expected_labels_len
    )
    tokenized_dataset = Dataset(tokenized_dataset.data.table.append_column("labels", labels_array))

    # Set format to PyTorch tensors for the columns we need
    # tokenized_dataset.set_format(
    #     "torch", columns=["input_ids", "attention_mask", "labels"]
    # )
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
    tokenized_dataset.set_format("torch", columns=format_columns)

    # Split based on is_valid column
    train_dataset = tokenized_dataset.filter(
        lambda example: not example["is_valid"], desc="Filtering train set"
    )
    test_dataset = tokenized_dataset.filter(
        lambda example: example["is_valid"], desc="Filtering test set"
    )

    # Remove the 'is_valid' column from the final datasets (optional)
    train_dataset = train_dataset.remove_columns("is_valid")
    test_dataset = test_dataset.remove_columns("is_valid")

    # Create a DatasetDict for train and test
    split_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    logging.info(f"The size of training set: {len(train_dataset)}")
    logging.info(f"The size of Evaluation set: {len(test_dataset)}")

    return (
        split_dataset,
        mlb.classes_,
        mlb,
        rare_labels,
        not_rare_labels,
        zero_labels,
    )  # Return split dataset and mlb for decoding


def load_label_partitions(base_dir):
    """Load label partitions and classes from labels_partition.json and compute indices."""
    # Usage example
    # base_dir = "/path/to/your/base_dir"
    # rare_labels_dict, not_rare_labels_dict, mlb_classes, rare_indices, not_rare_indices = load_label_partitions(base_dir)
    labels_file = os.path.join(base_dir, "labels_partition.json")
    with open(labels_file, "r") as f:
        labels_partition = json.load(f)

    rare_labels_dict = labels_partition["rare"]
    not_rare_labels_dict = labels_partition["not_rare"]
    zero_labels_dict = labels_partition.get("zero", {})  # may be absent in old partitions
    mlb_classes = labels_partition["classes"]

    # Compute indices from the loaded classes
    rare_indices = [i for i, cls in enumerate(mlb_classes) if cls in rare_labels_dict]
    not_rare_indices = [
        i for i, cls in enumerate(mlb_classes) if cls in not_rare_labels_dict
    ]
    zero_indices = [i for i, cls in enumerate(mlb_classes) if cls in zero_labels_dict]

    return (
        rare_labels_dict,
        not_rare_labels_dict,
        mlb_classes,
        rare_indices,
        not_rare_indices,
        zero_indices,
    )


# Function to load the base model and tokenizer
def load_base_model_and_tokenizer_too_old(
    model_name, hub_model_id, task_hub_model_id, hf_token
):
    """Load the Task 1 fine-tuned model and tokenizer."""
    # Initialize the Hugging Face API
    api = HfApi()

    # Get repository info for the hub_model_id
    repo_info = api.repo_info(repo_id=hub_model_id, token=hf_token)
    # Get repository info for the task_hub_model_id
    repo_info_task = api.repo_info(repo_id=task_hub_model_id, token=hf_token)

    # Extract the last modified timestamp for hub_model_id
    last_modified = repo_info.last_modified  # This is a datetime object
    last_modified_task = repo_info_task.last_modified  # This is a datetime object
    if last_modified and last_modified_task:
        last_modified_str = last_modified.strftime("%Y-%m-%d %H:%M:%S")
        last_modified_task_str = last_modified_task.strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"Model {hub_model_id} was last saved on: {last_modified_str}")
        logging.info(
            f"Model {task_hub_model_id} was last saved on: {last_modified_task_str}"
        )
    else:
        logging.info(
            f"Could not retrieve last saved timestamp for {hub_model_id} or {task_hub_model_id}"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, hub_model_id, token=hf_token)
    model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(
        hub_model_id, token=hf_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_base_model_and_tokenizer_old(
    hub_model_id="mistralai/Mistral-7B-Instruct-v0.3", lora_config=None, hf_token=None
):
    """
    Load a pre-trained or PEFT model and tokenizer, apply 4-bit quantization and LoRA if not PEFT.

    Args:
        hub_model_id (str): ID of the model to load from Hugging Face.
        lora_config (LoraConfig, optional): Custom LoRA configuration. If None, default is used.
        hf_token (str, optional): Hugging Face API token for private models.

    Returns:
        tuple: (model, tokenizer) - The quantized model (with LoRA or PEFT) and the tokenizer.

    Raises:
        ValueError: If hub_model_id is invalid.
        RuntimeError: If model, tokenizer, or PEFT/LoRA loading fails.
    """
    if not isinstance(hub_model_id, str) or not hub_model_id:
        raise ValueError("hub_model_id must be a non-empty string")

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
        logging.info(f"🔤 Loading tokenizer for model: {hub_model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(
            hub_model_id, token=hf_token, trust_remote_code=True
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            logging.info("📝 Setting pad token to eos token...")
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise RuntimeError(f"Tokenizer loading failed: {e}")

    # Check if the model is a PEFT model
    try:
        logging.info(f"🔍 Checking if {hub_model_id} is a PEFT model...")
        peft_config = PeftConfig.from_pretrained(hub_model_id, token=hf_token)
        is_peft_model = True
        base_model_name = peft_config.base_model_name_or_path
        logging.info(f"✅ Detected PEFT model. Base model: {base_model_name}")
    except Exception as e:
        logging.info(
            f"ℹ️ {hub_model_id} is not a PEFT model: {e}. Proceeding as base model."
        )
        is_peft_model = False
        base_model_name = hub_model_id

    # Load model configuration for type detection
    try:
        logging.info(f"🔍 Loading model configuration for {base_model_name}...")
        config = AutoConfig.from_pretrained(
            base_model_name, trust_remote_code=True, token=hf_token
        )
        model_type = config.model_type.lower()
        architectures = config.architectures if hasattr(config, "architectures") else []
        logging.info(f"Model type: {model_type}, Architectures: {architectures}")
    except Exception as e:
        logging.warning(f"Failed to load model config: {e}. Defaulting to causal LM.")
        model_type = "unknown"
        architectures = []

    # Map model types to task types and LoRA target modules
    model_type_mapping = {
        "bert": {
            "model_class": AutoModelForMaskedLM,
            "task_type": "MASKED_LM",
            "lora_targets": ["query", "key", "value", "output.dense"],
        },
        "roberta": {
            "model_class": AutoModelForMaskedLM,
            "task_type": "MASKED_LM",
            "lora_targets": ["query", "key", "value", "output.dense"],
        },
        "albert": {
            "model_class": AutoModelForMaskedLM,
            "task_type": "MASKED_LM",
            "lora_targets": ["query", "key", "value", "output.dense"],
        },
        "llama": {
            "model_class": AutoModelForCausalLM,
            "task_type": "CAUSAL_LM",
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "mistral": {
            "model_class": AutoModelForCausalLM,
            "task_type": "CAUSAL_LM",
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "phi3": {
            "model_class": AutoModelForCausalLM,
            "task_type": "CAUSAL_LM",
            # Phi-3-mini/medium fuse q/k/v into qkv_proj, unlike Llama/
            # Mistral's separate projections.
            "lora_targets": ["qkv_proj", "o_proj"],
        },
        "phi3small": {
            "model_class": AutoModelForCausalLM,
            "task_type": "CAUSAL_LM",
            # Phi-3-small (loaded via trust_remote_code from Microsoft's own
            # modeling_phi3_small.py) names its fused q/k/v and output
            # projections differently from phi3-mini/medium.
            "lora_targets": ["query_key_value", "dense"],
        },
        "gpt2": {
            "model_class": AutoModelForCausalLM,
            "task_type": "CAUSAL_LM",
            "lora_targets": ["c_attn", "c_proj"],
        },
    }

    # Determine model class and LoRA defaults
    if model_type in model_type_mapping:
        model_info = model_type_mapping[model_type]
    else:
        is_causal_lm = any("CausalLM" in arch for arch in architectures)
        if is_causal_lm:
            model_info = {
                "model_class": AutoModelForCausalLM,
                "task_type": "CAUSAL_LM",
                "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        else:
            logging.warning(
                f"Unknown model type '{model_type}'. Defaulting to causal LM with generic LoRA targets."
            )
            model_info = {
                "model_class": AutoModelForCausalLM,
                "task_type": "CAUSAL_LM",
                "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }

    # Load base model
    try:
        logging.info(f"🧠 Loading base model: {base_model_name}...")
        base_model = model_info["model_class"].from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )
    except Exception as e:
        logging.error(f"Failed to load base model: {e}")
        raise RuntimeError(f"Base model loading failed: {e}")

    # Handle PEFT model
    if is_peft_model:
        try:
            logging.info(f"🧩 Loading PEFT adapters for {hub_model_id}...")
            model = PeftModel.from_pretrained(base_model, hub_model_id, token=hf_token)
            logging.info("🔧 Before enabling PEFT adapters for training")
            log_print_output(func=model.print_trainable_parameters)
            # Ensure adapters are trainable
            model.train()  # Set model to training mode to enable adapter gradients
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "adapter" in name.lower():
                    param.requires_grad = True
            logging.info("🔧 After Enabling PEFT adapters for training")
            log_print_output(func=model.print_trainable_parameters)
            # Verify trainable parameters
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            if trainable_params == 0:
                logging.warning(
                    "No trainable parameters detected in PEFT model. Adapters may not be properly configured."
                )
        except Exception as e:
            logging.error(f"Failed to load PEFT adapters: {e}")
            raise RuntimeError(f"PEFT adapter loading failed: {e}")
    else:
        # Define default LoRA configuration if none provided
        if lora_config is None:
            logging.info("🔧 Setting up default LoRA configuration...")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=model_info["lora_targets"],
                lora_dropout=0.05,
                bias="none",
                task_type=model_info["task_type"],
            )

        # Log LoRA configuration
        logging.info(
            f"🔍 LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
            f"targets={lora_config.target_modules}, dropout={lora_config.lora_dropout}"
        )

        # Apply LoRA
        try:
            logging.info("🧩 Applying LoRA adapters to model...")
            model = get_peft_model(base_model, lora_config)
            model.print_trainable_parameters()
        except Exception as e:
            logging.error(f"Failed to apply LoRA: {e}")
            raise RuntimeError(f"LoRA application failed: {e}")

    logging.info("✅ Model and tokenizer successfully loaded!")
    return model, tokenizer


def load_base_model_and_tokenizer(
    hub_model_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_name=None,
    lora_config=None,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    hf_token=None,
    device_map="auto",
):
    """
    Load a pre-trained or PEFT model and tokenizer, apply 4-bit quantization and LoRA if not PEFT.

    Args:
        hub_model_id (str): ID (Hub id or local checkpoint dir) of the model to load.
        model_name (str, optional): Hub id of the original base model, used only
            for the tokenizer. Defaults to hub_model_id when omitted. hub_model_id
            is frequently a local dir (e.g. a previous stage's --skip_push_to_hub
            output) that never received the base model's custom tokenizer code
            (save_pretrained doesn't copy tokenization_*.py), so the tokenizer
            must be loaded from the true Hub id whenever the two diverge.
        lora_config (LoraConfig, optional): Custom LoRA configuration. If None, default is used.
        lora_r (int): LoRA rank for default config.
        lora_alpha (int): LoRA alpha for default config.
        lora_dropout (float): LoRA dropout for default config.
        hf_token (str, optional): Hugging Face API token for private models.
        device_map (str or dict): Device placement strategy (default: "auto").

    Returns:
        tuple: (model, tokenizer) - The quantized model (with LoRA or PEFT) and the tokenizer.

    Raises:
        ValueError: If hub_model_id is invalid.
        RuntimeError: If model, tokenizer, or PEFT/LoRA loading fails.
    """
    if not isinstance(hub_model_id, str) or not hub_model_id:
        raise ValueError("hub_model_id must be a non-empty string")

    tokenizer_source = model_name or hub_model_id

    logging.info("🚀 Starting model and tokenizer loading process...")

    # Define 4-bit quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    logging.info("📊 Quantization config: 4-bit, nf4, double_quant, bfloat16")

    # Load tokenizer
    try:
        logging.info(f"🔤 Loading tokenizer for model: {tokenizer_source}...")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, token=hf_token, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            logging.info("📝 Setting pad token to eos token...")
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise RuntimeError(f"Tokenizer loading failed: {e}")

    # Check if the model is a PEFT model
    try:
        logging.info(f"🔍 Checking if {hub_model_id} is a PEFT model...")
        peft_config = PeftConfig.from_pretrained(hub_model_id, token=hf_token)
        is_peft_model = True
        base_model_name = peft_config.base_model_name_or_path
        logging.info(f"✅ Detected PEFT model. Base model: {base_model_name}")
    except Exception:
        logging.info(f"ℹ️ {hub_model_id} is not a PEFT model. Proceeding as base model.")
        is_peft_model = False
        base_model_name = hub_model_id

    # Load model configuration for type detection
    try:
        logging.info(f"🔍 Loading model configuration for {base_model_name}...")
        config = AutoConfig.from_pretrained(
            base_model_name, trust_remote_code=True, token=hf_token
        )
    except Exception as e:
        logging.info(f"ℹ️ AutoConfig failed: {e}. Attempting LTRConfig...")
        try:
            config = LTRConfig.from_pretrained(
                base_model_name, trust_remote_code=True, token=hf_token
            )
        except Exception as e:
            logging.error(f"Failed to load model config with LTRConfig: {e}")
            raise RuntimeError(f"Model configuration loading failed for {base_model_name}: {e}")
    model_type = config.model_type.lower()
    architectures = config.architectures if hasattr(config, "architectures") else []
    logging.info(f"Model type: {model_type}, Architectures: {architectures}")

    # Map model types to task types and LoRA target modules
    model_type_mapping = {
        "bert": {
            "model_class": AutoModelForMaskedLM,
            "task_type": "MASKED_LM",
            "lora_targets": ["query", "key", "value", "output.dense"],
        },
        "roberta": {
            "model_class": AutoModelForMaskedLM,
            "task_type": "MASKED_LM",
            "lora_targets": ["query", "key", "value", "output.dense"],
        },
        "albert": {
            "model_class": AutoModelForMaskedLM,
            "task_type": "MASKED_LM",
            "lora_targets": ["query", "key", "value", "output.dense"],
        },
        "llama": {
            "model_class": AutoModelForCausalLM,
            "task_type": "CAUSAL_LM",
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "mistral": {
            "model_class": AutoModelForCausalLM,
            "task_type": "CAUSAL_LM",
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "phi3": {
            "model_class": AutoModelForCausalLM,
            "task_type": "CAUSAL_LM",
            # Phi-3-mini/medium fuse q/k/v into qkv_proj, unlike Llama/
            # Mistral's separate projections.
            "lora_targets": ["qkv_proj", "o_proj"],
        },
        "phi3small": {
            "model_class": AutoModelForCausalLM,
            "task_type": "CAUSAL_LM",
            # Phi-3-small (loaded via trust_remote_code from Microsoft's own
            # modeling_phi3_small.py) names its fused q/k/v and output
            # projections differently from phi3-mini/medium.
            "lora_targets": ["query_key_value", "dense"],
        },
        "gpt2": {
            "model_class": AutoModelForCausalLM,
            "task_type": "CAUSAL_LM",
            "lora_targets": ["c_attn", "c_proj"],
        },
        "ltr_model": {
            "model_class": LTRModel,
            "task_type": "PLANT_ATTENTION",
            "lora_targets": [],
        },
    }

    # Determine model class and LoRA defaults
    if model_type in model_type_mapping:
        model_info = model_type_mapping[model_type]
    else:
        is_causal_lm = any("CausalLM" in arch for arch in architectures)
        if is_causal_lm:
            model_info = {
                "model_class": AutoModelForCausalLM,
                "task_type": "CAUSAL_LM",
                "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        else:
            logging.warning(
                f"Unknown model type '{model_type}' with architectures {architectures}. "
                "Defaulting to causal LM, but this may lead to compatibility issues."
            )
            model_info = {
                "model_class": AutoModelForCausalLM,
                "task_type": "CAUSAL_LM",
                "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }

    # Set tokenizer padding side based on task type
    tokenizer.padding_side = "right" if model_info["task_type"] == "CAUSAL_LM" else "left"

    # Load base model
    try:
        logging.info(f"🧠 Loading base model: {base_model_name}...")
        # torch_dtype=bfloat16 below only governs *parameter* dtype. Buffers
        # built at construction time from torch.get_default_dtype() (e.g.
        # Phi-3-small's RotaryEmbedding cos/sin cache, which passes
        # dtype=None -> get_default_dtype()) are NOT covered by torch_dtype
        # and default to fp32, upcasting bf16 query/key through the rope
        # multiply -- the fp32 tensor that actually reaches flash-attn's
        # fp16/bf16-only kernel. Setting the process default dtype for the
        # duration of construction makes the cache bf16 from the start;
        # restored in `finally` so it doesn't leak into unrelated code.
        prev_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            base_model = model_info["model_class"].from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                token=hf_token,
                torch_dtype=torch.bfloat16,  # bnb_4bit_compute_dtype above only
                # governs the quantized linear layers; non-quantized submodules
                # (attention/hidden states) still default to fp32 without this,
                # which breaks custom fp16/bf16-only attention kernels (e.g.
                # Phi-3-small's block-sparse/flash-attn). Harmless for
                # Mistral/Llama.
            )
        finally:
            torch.set_default_dtype(prev_default_dtype)
        if hasattr(base_model, 'lookup_table'):
            logging.info("🔧 Removing lookup_table from model to save memory during training 💾")
            logging.info("📋 Original learning-to-rank model used it for ground_truth/loss computation")
            logging.info("⚙️ This standalone setup uses different loss mechanism - lookup_table not needed ✨")
            setattr(base_model, 'lookup_table', None)
    except Exception as e:
        logging.error(f"Failed to load base model: {e}")
        raise RuntimeError(f"Base model loading failed: {e}")

    # Handle PEFT model
    if is_peft_model:
        try:
            logging.info(f"🧩 Loading PEFT adapters for {hub_model_id}...")
            model = PeftModel.from_pretrained(base_model, hub_model_id, token=hf_token)
            logging.info("🔧 Before enabling PEFT adapters")
            log_print_output(func=model.print_trainable_parameters)
            # Ensure adapters are trainable
            model.train()
            trainable_params = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "adapter" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(name)
            if trainable_params:
                logging.info(f"Enabled gradients for parameters: {trainable_params}")
            logging.info("🔧 After enabling PEFT adapters")
            log_print_output(func=model.print_trainable_parameters)
            # Verify trainable parameters
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_count == 0:
                logging.warning("No trainable parameters in PEFT model. Check adapter config.")
        except Exception as e:
            logging.error(f"Failed to load PEFT adapters: {e}")
            raise RuntimeError(f"PEFT adapter loading failed: {e}")
    else:
        # Apply LoRA only if lora_targets are specified and model is not LTR type without PEFT
        if lora_config is None and model_info["lora_targets"]:
            logging.info("🔧 Setting up default LoRA configuration...")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=model_info["lora_targets"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=model_info["task_type"],
            )
            # Log LoRA configuration
            logging.info(
                f"🔍 LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
                f"targets={lora_config.target_modules}, dropout={lora_config.lora_dropout}"
            )
            # Apply LoRA
            try:
                logging.info("🧩 Applying LoRA adapters to model...")
                model = get_peft_model(base_model, lora_config)
                # PEFT initializes new LoRA A/B weights in fp32 regardless of
                # the base model's dtype -- same fix as learning2rank.py's
                # Stage 4 load; see that file's comment for the full
                # reasoning and verification against peft==0.20.0 +
                # bitsandbytes Linear4bit.
                model = model.to(torch.bfloat16)
                log_print_output(func=model.print_trainable_parameters)
            except Exception as e:
                logging.error(f"Failed to apply LoRA: {e}")
                raise RuntimeError(f"LoRA application failed: {e}")
        else:
            logging.info("ℹ️ No LoRA applied: either custom LoRA config provided or model type (e.g., LTR) does not support LoRA.")
            model = base_model

    logging.info("✅ Model and tokenizer successfully loaded!")
    return model, tokenizer

# Function to Define Training Arguments
def define_training_args(
    output_dir="./mistral_mimic4",
    task_hub_model_id="deb101/mistral-7b-instruct-v0.3-mimic4-adapt-multilabel-classify",
    hub_token=None,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-4,
    warmup_steps=500,
    metric_for_best_model: str = "f1_macro",
    push_to_hub: bool = True,
):
    """Define training arguments for the Trainer."""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        bf16=True,  # matches the bf16 model loads + bnb compute dtype +
        # accelerate mixed_precision=bf16 -- see learning2rank.py's
        # equivalent comment for the full reasoning.
        save_safetensors=False,  # Phi-3-small ties embed_tokens/lm_head
        # (tie_word_embeddings=True, unlike Mistral/Llama); safetensors
        # can't serialize tensors that share memory and Trainer._save
        # crashes with "Some tensors share memory" for the underlying
        # base_model whenever save_safetensors is left at its True
        # default. torch.save (pytorch_model.bin) handles aliased
        # tensors natively. Harmless no-op for untied Mistral/Llama.
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


def get_optimizer_and_scheduler(
    model,
    train_dataset,
    training_args,
    initial_lr=1e-4,
    final_lr=1e-6,
    warmup_steps=500,
):
    """
    Creates an optimizer and scheduler with linear warmup and decay.

    Args:
        model: The model to optimize (e.g., Mistral-7B).
        train_dataset: The training dataset (e.g., your MIMIC-IV dataset).
        training_args: TrainingArguments object with training config.
        initial_lr (float): Initial learning rate after warmup (default: 1e-4).
        final_lr (float): Final learning rate at the end of training (default: 1e-6).
        warmup_steps (int): Number of warmup steps (e.g., 500).

    Returns:
        tuple: (optimizer, scheduler) to pass to Trainer as optimizers=.
    """
    # Compute num_training_steps
    total_training_samples = len(train_dataset)  # e.g., 102,477 from your log
    batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    num_train_epochs = training_args.num_train_epochs

    # Base calculation
    num_training_steps = (
        total_training_samples // (batch_size * gradient_accumulation_steps)
    ) * num_train_epochs

    # Adjust for max_steps if set
    if training_args.max_steps > 0:
        num_training_steps = min(num_training_steps, training_args.max_steps)

    # Define optimizer with initial_lr
    optimizer = AdamW(model.parameters(), lr=initial_lr)
    # Define optimizer with parameter groups and different initial LRs
    # optimizer = AdamW([
    #     # {"params": [p for n, p in model.named_parameters() if p.requires_grad and "base" in n.lower() and "lora" not in n.lower()], "lr": 1e-6},  # Unfrozen base layers
    #     # {"params": [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()], "lr": 1e-5},              # Unfrozen LoRA layers
    #     {"params": [p for n, p in model.named_parameters() if p.requires_grad and "base" not in n.lower()], "lr": initial_lr}    # Custom layers use initial_lr
    # ])

    # Base linear scheduler
    # base_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=num_training_steps
    # )

    # Use cosine_with_min_lr scheduler
    scheduler = get_scheduler(
        name="cosine_with_min_lr",  # or SchedulerType.COSINE_WITH_MIN_LR
        # name="cosine_with_restarts",  # or SchedulerType.COSINE_WITH_MIN_LR
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs={"min_lr": final_lr},  # Set final_lr as min_lr
        # scheduler_specific_kwargs={"num_cycles": 3}  # Set final_lr as min_lr
    )

    # Custom lambda to enforce final_lr
    # def lr_lambda(current_step):
    #     base_lr = base_scheduler.get_last_lr()[0]
    #     if current_step < warmup_steps:
    #         return base_lr / initial_lr
    #     else:
    #         progress = (current_step - warmup_steps) / (num_training_steps - warmup_steps)
    #         return 1.0 - progress * (1.0 - final_lr / initial_lr)

    # Wrap with LambdaLR
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # import pdb; pdb.set_trace()

    return (optimizer, scheduler)


# =============================================================================
# CallBacks and CustomTrainer
# =============================================================================
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


# Custom callback for logging
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_local_process_zero:
            # Only log as "Training metrics" if 'eval_' keys are absent
            if not any(k.startswith("eval_") for k in logs.keys()):
                log_message = "Training metrics: " + ", ".join(
                    f"{k}: {v}" for k, v in logs.items()
                )
                # log_message = "Training metrics:\n" + "\n".join(f"{k}: {v}" for k, v in logs.items())
                logging.info(log_message)

        # if logs is not None:
        #     log_message = "Training metrics: " + ", ".join(f"{k}: {v}" for k, v in logs.items())
        #     logging.info.info(log_message)

    def on_step_end_old(self, args, state, control, logs=None, **kwargs):
        # Only log every logging_steps and on local process zero
        if state.global_step % 12 == 11 and state.is_local_process_zero:
            # Access optimizer and scheduler from kwargs
            optimizer = kwargs["optimizer"]
            scheduler = kwargs["lr_scheduler"]

            # Get learning rates for each parameter group
            lrs = scheduler.get_last_lr()  # List of LRs for each group
            import pdb

            pdb.set_trace()

            # Compute gradient norms and track non-empty groups
            grad_norms = []
            group_names = ["base", "lora", "custom"]  # Matches optimizer group order
            active_groups = []
            for group_idx, param_group in enumerate(optimizer.param_groups):
                if len(param_group["params"]) == 0:  # Skip empty groups
                    continue
                group_norm = 0.0
                for param in param_group["params"]:
                    if (
                        param.grad is not None
                    ):  # Should always be true here for requires_grad=True
                        group_norm += torch.norm(param.grad, p=2).item() ** 2
                grad_norms.append(group_norm**0.5 if group_norm > 0 else 0.0)
                active_groups.append(group_idx)

            # Construct log message with metrics only for non-empty groups
            log_message = ["Training metrics:"]
            log_message.append(f"loss: {logs.get('loss', 'N/A') if logs else 'N/A'}")

            for idx, group_idx in enumerate(active_groups):
                log_message.append(
                    f"{group_names[group_idx]}_grad_norm: {grad_norms[idx]:.4f}"
                )
            for idx, group_idx in enumerate(active_groups):
                log_message.append(
                    f"{group_names[group_idx]}_lr: {lrs[group_idx]:.10f}"
                )

            log_message.append(
                f"epoch: {state.epoch if state.epoch is not None else 'N/A'}"
            )
            logging.info(", ".join(log_message))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if (
            metrics is not None and state.is_local_process_zero
        ):  # Only log on main process
            # log_message = "Evaluation metrics: " + ", ".join(f"{k}: {v}" for k, v in metrics.items())
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


class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Separate 'text' and other features
        text_features = (
            [f.pop("text") for f in features] if "text" in features[0] else None
        )
        # labels_features = (
        #     [f.pop("labels") for f in features] if "labels" in features[0] else None
        # )

        # Let DataCollatorWithPadding handle the rest
        batch = super().__call__(features)

        # Optionally, add 'text' and 'labels' back to the batch if needed
        # if text_features is not None:
        #     batch["text"] = text_features
        # if labels_features is not None:
        #     batch["labels"] = labels_features

        return batch


# Custom Trainer
class CustomTrainer(Trainer):

    def __init__(
        self,
        *args,
        tokenizer: AutoTokenizer = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if tokenizer:
            self.tokenizer = tokenizer
            logging.info(
                f"We are registering the tokenizer {tokenizer.name_or_path} in Custom Trainer"
            )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to process in batches and avoid GPU memory overload."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("No eval_dataset provided.")

        # Remove 'token_type_ids' from the dataset if it exists
        if "token_type_ids" in eval_dataset.column_names:
            logging.info(
                "Removing 'token_type_ids' from eval_dataset as they are not needed."
            )
            eval_dataset = eval_dataset.remove_columns(["token_type_ids"])

        # eval_dataloader_old = DataLoader(
        #     eval_dataset, batch_size=self.args.per_device_eval_batch_size, shuffle=False
        # )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=CustomDataCollator(tokenizer=self.processing_class),
        )

        # "candidate" mode's logits/labels columns mean a different real label
        # per row (candidate_label_ids[row, k]), not a fixed global label index
        # -- collect those columns too so they can be remapped before scoring,
        # see remap_candidate_eval_arrays.
        is_candidate_mode = "candidate_label_ids" in eval_dataset.column_names

        self.model.eval()
        all_logits = []
        all_labels = []
        all_candidate_label_ids = []
        all_candidate_mask = []
        total_loss = 0.0
        num_samples = 0

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
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                if "loss" in outputs:
                    total_loss += outputs["loss"].item() * batch["input_ids"].size(0)

                all_logits.append(outputs["logits"].cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())
                if is_candidate_mode:
                    all_candidate_label_ids.append(batch["candidate_label_ids"].cpu().numpy())
                    all_candidate_mask.append(batch["candidate_mask"].cpu().numpy())
                num_samples += batch["input_ids"].size(0)

                del outputs, batch
                torch.cuda.empty_cache()

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        eval_logits = np.concatenate(all_logits)
        eval_labels = np.concatenate(all_labels)
        if is_candidate_mode:
            eval_logits, eval_labels = remap_candidate_eval_arrays(
                eval_logits, eval_labels,
                np.concatenate(all_candidate_label_ids),
                np.concatenate(all_candidate_mask),
                self.model.num_labels,
            )
        metrics = self.compute_metrics((eval_logits, eval_labels))
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

    def _save_old(self, output_dir=None, state_dict=None):
        # Use the provided output_dir or default to current checkpoint dir
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save the model weights in safetensors format
        if state_dict is None:
            state_dict = self.model.state_dict()
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
        logging.info(f"Model weights saved in safetensors format: {output_dir}")

        # Explicitly save the config
        if hasattr(self.model, "config") and self.model.config is not None:
            self.model.config.save_pretrained(output_dir)
            logging.info(f"Config saved in checkpoint: {output_dir}")
        else:
            logging.warning("Model has no config to save in checkpoint")

        # Save training args
        self._save_training_args(output_dir)

    def _save(self, output_dir=None, state_dict=None):
        # Use the provided output_dir or default to current checkpoint dir
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save the model weights, format gated on self.args.save_safetensors.
        # This override bypasses Trainer's own save entirely (that's the
        # point of overriding _save), so it must honor the flag itself --
        # setting save_safetensors=False on TrainingArguments alone did
        # nothing here. safetensors can't serialize Phi-3-small's tied
        # embed_tokens/lm_head (tie_word_embeddings=True, unlike
        # Mistral/Llama); torch.save handles aliased tensors natively.
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

        # Log saved files in a table with file sizes
        saved_files = [
            (f, os.path.getsize(os.path.join(output_dir, f)) / 1024**2)
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ]
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

        if not baseline_improved:
            logging.info(
                "Baseline was not improved during training, keeping original model"
            )
            return  # Skip loading, keep the original model

        # Proceed with default loading logic if baseline improved
        logging.info(f"Loading best model from {self.state.best_model_checkpoint}")
        if self.state.best_model_checkpoint and os.path.exists(
            self.state.best_model_checkpoint
        ):
            weights_file = os.path.join(
                self.state.best_model_checkpoint, "model.safetensors"
            )
            if os.path.isfile(weights_file):
                state_dict = load_file(weights_file)
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
    k_values=[1, 5, 8, 15, 25],
    rare_labels=None,
    not_rare_labels=None,
    zero_labels=None,
    eval=False,
    final_lr_scheduling=1e-6,
    label_space="label",
):
    """Train or evaluate the multilabel classification model based on eval flag.

    `label_space="cluster"` additionally reports router_coverage_at_k
    (Phase 7 of the full-scale candidate-generator plan, see
    README_AMAZON3M.md) -- a per-document all-or-nothing check of whether
    the top-k predicted clusters' member labels contain every one of that
    document's true labels, on top of the ordinary cluster-level
    precision/recall/F1/PSP metrics already computed below.
    """

    # Function to compute metrics
    def compute_metrics_basic(eval_pred):
        """Compute F1, Precision@k, and Recall@k for multilabel classification at k=5, 8, 15, 25."""
        logits, labels = (
            eval_pred  # logits: (n_samples, num_labels), labels: (n_samples, num_labels)
        )
        if isinstance(logits, tuple):
            logits = logits[0]

        # Convert logits to probabilities and predictions
        probs = torch.sigmoid(
            torch.tensor(logits, dtype=torch.float32)
        ).numpy()  # Shape: (n_samples, num_labels)
        labels = torch.tensor(
            labels, dtype=torch.int32
        ).numpy()  # Shape: (n_samples, num_labels)
        predictions = (probs > 0.5).astype(int)  # Binary predictions for F1

        # Compute F1 scores
        f1_micro = f1_score(labels, predictions, average="micro")
        f1_macro = f1_score(labels, predictions, average="macro")

        # Compute Precision@k and Recall@k for each k
        metrics = {"f1_micro": f1_micro, "f1_macro": f1_macro}
        inv_w = get_inv_propensity(labels)
        for k in k_values:
            # Get top-k predicted labels per sample
            top_k_indices = np.argsort(probs, axis=1)[:, -k:]  # Shape: (n_samples, k)
            top_k_preds = np.zeros_like(labels)  # Shape: (n_samples, num_labels)
            for i in range(len(labels)):
                top_k_preds[i, top_k_indices[i]] = 1  # Set top-k indices to 1

            # Precision@k: (# true positives in top-k) / k
            true_positives_at_k = np.sum(
                top_k_preds * labels, axis=1
            )  # Shape: (n_samples,)
            precision_at_k = np.mean(true_positives_at_k / k)

            # Recall@k: (# true positives in top-k) / (# true labels)
            true_labels_per_sample = np.sum(labels, axis=1)  # Shape: (n_samples,)
            recall_at_k = np.mean(
                np.where(
                    true_labels_per_sample > 0,
                    true_positives_at_k / true_labels_per_sample,
                    0,
                )
            )

            # Add to metrics dict with k-specific keys
            metrics[f"precision_at_{k}"] = precision_at_k
            metrics[f"recall_at_{k}"] = recall_at_k
            metrics[f"psp_at_{k}"] = get_psp_at_k(probs, labels, inv_w, k)

        return metrics

    # Function to compute metrics for rare and not_rare label class
    def compute_metrics_with_labels(eval_pred, rare_labels, not_rare_labels, zero_labels=None):
        """Compute F1, Precision@k, and Recall@k for multilabel classification at k=1, 5, 8, 15, 25,
        plus per-bucket metrics for rare/non-rare and for the train-frequency
        buckets zero-shot / few-shot / head (see build_label_or_cluster_vectors)."""
        logits, labels = (
            eval_pred  # logits: (n_samples, num_labels), labels: (n_samples, num_labels)
        )
        if isinstance(logits, tuple):
            logits = logits[0]

        # Convert logits to probabilities and predictions
        probs = torch.sigmoid(
            torch.tensor(logits, dtype=torch.float32)
        ).numpy()  # Shape: (n_samples, num_labels)
        labels = torch.tensor(
            labels, dtype=torch.int32
        ).numpy()  # Shape: (n_samples, num_labels)
        predictions = (probs > 0.5).astype(int)  # Binary predictions for F1

        # Compute F1 scores (overall)
        f1_micro = f1_score(labels, predictions, average="micro")
        f1_macro = f1_score(labels, predictions, average="macro")

        # Compute Precision@k and Recall@k for each k (overall)
        metrics = {"f1_micro": f1_micro, "f1_macro": f1_macro}
        inv_w = get_inv_propensity(labels)
        for k in k_values:
            top_k_indices = np.argsort(probs, axis=1)[:, -k:]  # Shape: (n_samples, k)
            top_k_preds = np.zeros_like(labels)  # Shape: (n_samples, num_labels)
            for i in range(len(labels)):
                top_k_preds[i, top_k_indices[i]] = 1  # Set top-k indices to 1

            true_positives_at_k = np.sum(
                top_k_preds * labels, axis=1
            )  # Shape: (n_samples,)
            precision_at_k = np.mean(true_positives_at_k / k)
            true_labels_per_sample = np.sum(labels, axis=1)  # Shape: (n_samples,)
            recall_at_k = np.mean(
                np.where(
                    true_labels_per_sample > 0,
                    true_positives_at_k / true_labels_per_sample,
                    0,
                )
            )

            metrics[f"precision_at_{k}"] = precision_at_k
            metrics[f"recall_at_{k}"] = recall_at_k
            metrics[f"psp_at_{k}"] = get_psp_at_k(probs, labels, inv_w, k)

        # Metrics for rare labels. Guard against an empty partition (possible
        # on small/sample datasets where no label clears rare_threshold, or
        # every label does) -- sklearn's f1_score/precision_recall_fscore_support
        # raise ValueError("unknown is not supported") on a 0-column target.
        if len(rare_labels) > 0:
            rare_preds = predictions[:, rare_labels]
            rare_labels_true = labels[:, rare_labels]
            rare_f1_micro = f1_score(
                rare_labels_true, rare_preds, average="micro", zero_division=0
            )
            rare_f1_macro = f1_score(
                rare_labels_true, rare_preds, average="macro", zero_division=0
            )
            rare_precision, rare_recall, rare_f1, _ = precision_recall_fscore_support(
                rare_labels_true, rare_preds, average="micro", zero_division=0
            )
        else:
            rare_labels_true = None
            rare_f1_micro = rare_f1_macro = rare_precision = rare_recall = 0.0
        metrics["rare_f1_micro"] = (
            rare_f1_micro  # Using f1_score for consistency with original
        )
        metrics["rare_f1_macro"] = (
            rare_f1_macro  # Using f1_score for consistency with original
        )
        metrics["rare_precision"] = rare_precision
        metrics["rare_recall"] = rare_recall
        # metrics["rare_f1"] = rare_f1  # From precision_recall_fscore_support

        # Precision@k and Recall@k for rare labels
        for k in k_values:
            if len(rare_labels) > 0:
                rare_probs = probs[:, rare_labels]
                rare_top_k_indices = np.argsort(rare_probs, axis=1)[:, -k:]
                rare_top_k_preds = np.zeros_like(rare_labels_true)
                for i in range(len(rare_labels_true)):
                    rare_top_k_preds[i, rare_top_k_indices[i]] = 1

                rare_true_positives_at_k = np.sum(
                    rare_top_k_preds * rare_labels_true, axis=1
                )
                rare_precision_at_k = np.mean(rare_true_positives_at_k / k)
                rare_true_labels_per_sample = np.sum(rare_labels_true, axis=1)
                rare_recall_at_k = np.mean(
                    np.where(
                        rare_true_labels_per_sample > 0,
                        rare_true_positives_at_k / rare_true_labels_per_sample,
                        0,
                    )
                )
            else:
                rare_precision_at_k = rare_recall_at_k = 0.0

            metrics[f"rare_precision_at_{k}"] = rare_precision_at_k
            metrics[f"rare_recall_at_{k}"] = rare_recall_at_k

        # Metrics for not rare labels -- same empty-partition guard as above.
        if len(not_rare_labels) > 0:
            not_rare_preds = predictions[:, not_rare_labels]
            not_rare_labels_true = labels[:, not_rare_labels]
            not_rare_f1_micro = f1_score(
                not_rare_labels_true, not_rare_preds, average="micro", zero_division=0
            )
            not_rare_f1_macro = f1_score(
                not_rare_labels_true, not_rare_preds, average="macro", zero_division=0
            )
            not_rare_precision, not_rare_recall, not_rare_f1, _ = (
                precision_recall_fscore_support(
                    not_rare_labels_true, not_rare_preds, average="micro", zero_division=0
                )
            )
        else:
            not_rare_labels_true = None
            not_rare_f1_micro = not_rare_f1_macro = not_rare_precision = not_rare_recall = 0.0
        metrics["not_rare_f1_micro"] = not_rare_f1_micro
        metrics["not_rare_f1_macro"] = not_rare_f1_macro
        metrics["not_rare_precision"] = not_rare_precision
        metrics["not_rare_recall"] = not_rare_recall
        # metrics["not_rare_f1"] = not_rare_f1

        # Precision@k and Recall@k for not rare labels
        for k in k_values:
            if len(not_rare_labels) > 0:
                not_rare_probs = probs[:, not_rare_labels]
                not_rare_top_k_indices = np.argsort(not_rare_probs, axis=1)[:, -k:]
                not_rare_top_k_preds = np.zeros_like(not_rare_labels_true)
                for i in range(len(not_rare_labels_true)):
                    not_rare_top_k_preds[i, not_rare_top_k_indices[i]] = 1

                not_rare_true_positives_at_k = np.sum(
                    not_rare_top_k_preds * not_rare_labels_true, axis=1
                )
                not_rare_precision_at_k = np.mean(not_rare_true_positives_at_k / k)
                not_rare_true_labels_per_sample = np.sum(not_rare_labels_true, axis=1)
                not_rare_recall_at_k = np.mean(
                    np.where(
                        not_rare_true_labels_per_sample > 0,
                        not_rare_true_positives_at_k / not_rare_true_labels_per_sample,
                        0,
                    )
                )
            else:
                not_rare_precision_at_k = not_rare_recall_at_k = 0.0

            metrics[f"not_rare_precision_at_{k}"] = not_rare_precision_at_k
            metrics[f"not_rare_recall_at_{k}"] = not_rare_recall_at_k

        # --- zero-shot / few-shot / head buckets (by TRAIN-only frequency) ---
        # zero = 0 training examples (a subset of rare); few = rare minus zero;
        # head = not_rare. Same per-note averaging convention as rare/not_rare
        # above (mean over ALL eval notes, contributing 0 for notes that have no
        # true code in the bucket). Read `<bucket>_precision_at_<median-k>` as the
        # headline per README (zero~@1-2, few~@5, head~@8, overall~@15).
        _zset = set(zero_labels) if zero_labels is not None else set()
        _buckets = (
            ("zero", sorted(_zset)),
            ("few", [i for i in rare_labels if i not in _zset]),
            ("head", list(not_rare_labels)),
        )
        for _bname, _bidx in _buckets:
            _has = len(_bidx) > 0
            if _has:
                _btrue = labels[:, _bidx]
                _bprobs = probs[:, _bidx]
                _bcnt = np.sum(_btrue, axis=1)
            for k in k_values:
                if _has:
                    _btopk = np.argsort(_bprobs, axis=1)[:, -k:]
                    _bpred = np.zeros_like(_btrue)
                    for i in range(len(_btrue)):
                        _bpred[i, _btopk[i]] = 1
                    _btp = np.sum(_bpred * _btrue, axis=1)
                    _bprec = float(np.mean(_btp / k))
                    _brec = float(np.mean(np.where(_bcnt > 0, _btp / _bcnt, 0)))
                else:
                    _bprec = _brec = 0.0
                metrics[f"{_bname}_precision_at_{k}"] = _bprec
                metrics[f"{_bname}_recall_at_{k}"] = _brec
            if _has:
                metrics[f"{_bname}_f1_micro"] = f1_score(
                    _btrue, predictions[:, _bidx], average="micro", zero_division=0
                )
                metrics[f"{_bname}_f1_macro"] = f1_score(
                    _btrue, predictions[:, _bidx], average="macro", zero_division=0
                )
            else:
                metrics[f"{_bname}_f1_micro"] = metrics[f"{_bname}_f1_macro"] = 0.0

        return metrics

    def compute_router_metrics(eval_pred):
        """Router-only diagnostic (label_space="cluster"): does the union of
        the top-k predicted clusters' member labels contain EVERY one of a
        document's true labels? A doc's true clusters are exactly the
        clusters marked 1 in its cluster multi-hot target (see
        build_label_or_cluster_vectors's "cluster" branch: cluster c is 1 iff
        it contains >=1 of the doc's true labels) -- so this reduces to
        checking those true-cluster columns are a subset of the top-k
        predicted columns. This is a per-document all-or-nothing check,
        unlike the label-averaged recall_at_k above: missing even one true
        cluster loses every label inside it, so this upper-bounds what the
        downstream fine stage could possibly recover once labels outside the
        shortlisted clusters are dropped from candidate generation (see
        README_AMAZON3M.md, Phase 7 of the full-scale candidate-generator
        plan).
        """
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
        labels = torch.tensor(labels, dtype=torch.int32).numpy()

        metrics = {}
        for k in k_values:
            top_k_indices = np.argsort(probs, axis=1)[:, -k:]
            top_k_preds = np.zeros_like(labels)
            for i in range(len(labels)):
                top_k_preds[i, top_k_indices[i]] = 1
            covered = np.all(labels <= top_k_preds, axis=1)
            metrics[f"router_coverage_at_{k}"] = float(covered.mean())
        return metrics

    # Ensure training_args is a TrainingArguments object
    if not isinstance(training_args, TrainingArguments):
        training_args = TrainingArguments(**training_args)

    if rare_labels is not None and not_rare_labels is not None:
        base_compute_metrics = lambda eval_pred: compute_metrics_with_labels(
            eval_pred, rare_labels, not_rare_labels, zero_labels
        )
    else:
        base_compute_metrics = compute_metrics_basic

    if label_space == "cluster":
        def compute_metrics(eval_pred):
            metrics = base_compute_metrics(eval_pred)
            metrics.update(compute_router_metrics(eval_pred))
            return metrics
    else:
        compute_metrics = base_compute_metrics

    # Get optimizer and scheduler
    optimizers = get_optimizer_and_scheduler(
        model=model,
        train_dataset=train_dataset,
        training_args=training_args,
        initial_lr=training_args.learning_rate,
        final_lr=final_lr_scheduling,
        warmup_steps=training_args.warmup_steps,
    )

    callbacks = [PrettyLoggingCallback(), ColoredProgressCallback()]
    if not eval:
        callbacks.append(
            PlotLossCallback()
        )  # Add PlotLossCallback only if not in eval mode

    # Initialize CustomTrainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
    )

    # If eval flag is True, skip training and simply return evaluation metrics
    if eval:
        logging.info("Evaluation mode: Skipping training, performing evaluation only")
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
    trainer.train()

    # Check if baseline was improved and adjust loading behavior
    # if training_args.load_best_model_at_end and not baseline_callback.get_baseline_improved():
    #     logging.info("No improvement over baseline during training. Skipping load_best_model_at_end to keep original model.")
    #     trainer.state.best_model_checkpoint = None  # Prevent loading non-existent checkpoint
    #     trainer.model = model  # Explicitly revert to original model (though not strictly needed)

    return trainer


# Function to save and push to hub
def save_and_push_old(
    trainer, tokenizer, output_dir, commit_message="Trained ICD10 Classifier"
):
    """Save model and tokenizer locally and push to Hub."""
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Model and tokenizer saved to: {output_dir}")

    trainer.push_to_hub(commit_message=commit_message)
    tokenizer.push_to_hub(trainer.args.hub_model_id, token=trainer.args.hub_token)
    print(f"Model and tokenizer pushed to Hub: {trainer.args.hub_model_id}")


# Function to save and push to hub
def save_and_push_old(
    trainer, tokenizer, output_dir, commit_message="Trained ICD10 Classifier"
):
    """Save custom layers in safetensors format and tokenizer locally, then push to Hub (root only)."""
    # Calculate and log the size of custom layers
    custom_params, _ = trainer.model.compute_trainable_params()
    custom_params_old = sum(
        p.numel() for n, p in trainer.model.named_parameters() if "base_model" not in n
    )
    custom_size_gb = custom_params * 4 / (1024**3)  # Assuming FP32
    logging.info(f"Custom layers parameters: {custom_params:,}")
    logging.info(f"Custom layers size: {custom_size_gb:.2f} GB")

    # Verify config exists
    if not hasattr(trainer.model, "config"):
        logging.error("Model has no config attribute!")
    else:
        logging.info(f"Config type: {type(trainer.model.config)}")

    # Check if root weights match the best checkpoint
    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint and os.path.exists(best_checkpoint):
        # Get current model custom weights
        current_custom_weights = {
            k: v for k, v in trainer.model.state_dict().items() if "base_model" not in k
        }
        logging.info(
            f"Extracted {len(current_custom_weights)} custom weights from current model"
        )

        # Determine the device of the current model
        model_device = next(trainer.model.parameters()).device
        logging.info(f"Current model device: {model_device}")

        # Load best checkpoint weights
        checkpoint_weights = load_file(
            os.path.join(best_checkpoint, "model.safetensors")
        )
        checkpoint_custom_weights = {
            k: v for k, v in checkpoint_weights.items() if "base_model" not in k
        }

        # Move checkpoint weights to the same device as the current model
        checkpoint_custom_weights = {
            k: v.to(model_device) for k, v in checkpoint_custom_weights.items()
        }
        logging.info("Moved checkpoint weights to model device")

        # Compare weights
        weights_match = all(
            k in checkpoint_custom_weights
            and torch.equal(current_custom_weights[k], checkpoint_custom_weights[k])
            for k in current_custom_weights
        )
        if weights_match:
            logging.info(
                f"Root model custom weights match best checkpoint: {best_checkpoint}"
            )
        else:
            logging.warning(
                f"Root model custom weights DO NOT match best checkpoint: {best_checkpoint}"
            )
    else:
        logging.warning("No best checkpoint found or accessible")

    # Save only custom parameters in safetensors format
    custom_state_dict = {
        k: v for k, v in trainer.model.state_dict().items() if "base_model" not in k
    }
    # Save parameters from unfrozen base, unfrozen LoRA, and custom layers in safetensors format
    # custom_state_dict = {
    #     k: v for k, v in trainer.model.state_dict().items()
    #     if any(p.equal(v) for group in trainer.optimizer.param_groups for p in group["params"])
    # }
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(custom_state_dict, safetensors_path)

    # Save full config with all attributes
    config_dict = trainer.model.config.to_dict()  # Get base config as dict
    # Ensure custom attributes are included (redundant but explicit)
    config_dict.update(
        {
            "num_labels": trainer.model.config.num_labels,
            "model_type": "multilabel_icd_classifier",
            "_name_or_path": trainer.model.config._name_or_path,
            "label_embedding_init": "randn",
            "attention_num_heads": 8,
            "attention_batch_first": True,
            "boost_mul_init": "ones",
            "boost_add_init": "zeros",
            "pooling_type": "adaptive_avg_pool1d",
            "pooling_output_size": trainer.model.config.pooling_output_size,
            "number_of_layers_of_base_unfrozen": trainer.model.config.num_layers_to_unfreeze,
        }
    )
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    logging.info("Full config saved successfully")

    # Save tokenizer
    try:
        tokenizer.save_pretrained(output_dir)
        logging.info("Tokenizer saved successfully")
    except Exception as e:
        logging.error(f"Failed to save tokenizer: {e}")

    logging.info(f"Custom model (safetensors) and tokenizer saved to: {output_dir}")
    logging.info(f"Files in output_dir: {os.listdir(output_dir)}")

    # Push to Hub (root files only)
    if trainer.args.hub_model_id and trainer.args.hub_token:
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(
                f"Load Mistral-7B from {trainer.model.config._name_or_path} and combine with these custom weights."
            )
        api = HfApi()
        root_files = [
            f
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ]
        for file_name in root_files:
            file_path = os.path.join(output_dir, file_name)
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=trainer.args.hub_model_id,
                token=trainer.args.hub_token,
                commit_message=commit_message,
            )
        logging.info(
            f"Custom model and tokenizer pushed to Hub: {trainer.args.hub_model_id}"
        )
    else:
        logging.info("Skipping push to Hub: hub_model_id or hub_token not set")


def save_and_push(
    trainer,
    tokenizer,
    output_dir,
    logfile_path,
    commit_message="Trained classifier model on MIMIC-IV",
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

        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        logging.info(f"🖌️ Tokenizer saved to: {output_dir}")

        # Copy log file into the output directory
        if os.path.isfile(logfile_path):
            log_dst_path = os.path.join(output_dir, os.path.basename(logfile_path))
            shutil.copy2(logfile_path, log_dst_path)
            logging.info(f"📜 Log file saved to: {log_dst_path}")
        else:
            logging.warning(f"⚠️ Provided log file not found: {logfile_path}")

        # Log saved files in a tabular format with file sizes
        saved_files = [
            (f, os.path.getsize(os.path.join(output_dir, f)) / 1024**2)
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ]
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

        # Verify critical files before pushing. Weights file: model.safetensors
        # or pytorch_model.bin -- CustomTrainer._save writes the latter
        # whenever save_safetensors=False (Phi-3-small's tied
        # embed_tokens/lm_head can't go through safetensors).
        has_weights_file = os.path.isfile(
            os.path.join(output_dir, "model.safetensors")
        ) or os.path.isfile(os.path.join(output_dir, "pytorch_model.bin"))
        missing_files = [] if has_weights_file else ["model.safetensors/pytorch_model.bin"]
        if not os.path.isfile(os.path.join(output_dir, "config.json")):
            missing_files.append("config.json")
        if missing_files:
            logging.error(f"❌ Missing required files: {missing_files}")
            raise FileNotFoundError(f"Missing required files: {missing_files}")

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


def load_custom_task_model_and_tokenizer_old(
    output_dir,
    hub_model_id,
    task_hub_model_id,
    model_exists_locally,
    do_fresh_training,
    device="cpu",
    hf_token=None,
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
    if config_dict.get("model_type") != "multilabel_icd_classifier":
        logging.warning(
            f"Config model_type is {config_dict.get('model_type')}, expected 'multilabel_icd_classifier'"
        )

    # Extract key config values
    num_labels = config_dict.get("num_labels")
    base_model_path = config_dict.get(
        "_name_or_path", "mistralai/Mistral-7B-Instruct-v0.3"
    )  # Default if missing
    if not num_labels:
        logging.error("Config missing 'num_labels' field")
        raise ValueError("Config must specify 'num_labels'")

    # Load the domain-adapted LLM first
    base_model, _ = load_base_model_and_tokenizer(
        base_model_path, hub_model_id, task_hub_model_id, hf_token
    )

    # Initialize custom model with the multi-label classification head
    model = define_model(base_model, num_labels)
    logging.info(f"Initialized MultilabelICDClassifier with {num_labels} labels")

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
    hub_model_id,
    task_hub_model_id,
    model_exists_locally,
    do_fresh_training,
    device="cpu",
    hf_token=None,
    skip_push_to_hub=False,
    model_name=None,
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
        **kwargs: Additional arguments if needed for model initialization.

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
            if not is_valid_hf_model(task_hub_model_id):
                logging.error(
                    f"💥 Could not find valid hf repo for this task, make sure if fresh train the model for this task before performing checkpoint restart"
                )
                raise ValueError(f"Invalid Hugging Face model ID: {task_hub_model_id}")
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

        if not num_labels:
            logging.error(f"❌ Config missing 'num_labels' field")
            raise ValueError("Config must specify 'num_labels'")
        logging.info(f"🔢 Num labels: {num_labels}, Base model path: {hub_model_id}")

        # Load the domain-adapted base model
        base_model, _ = load_base_model_and_tokenizer(hub_model_id, hf_token=hf_token)
        logging.info(f"🤖 Loaded base model from: {hub_model_id}")

        # Initialize custom model with the multi-label classification head. Reuse
        # whatever label_embed_rank the checkpoint was originally saved with (None
        # if it wasn't factorized) -- reconstructing with a different value would
        # give a dense/factorized architecture mismatch against the saved
        # state_dict's actual tensor shapes.
        # if "some_additional_arg" not in kwargs:
        #     logging.error(f"❌ some_additional_arg not found in kwargs")
        #     raise ValueError("some_additional_arg not found in kwargs")
        # some_additional_arg = kwargs["some_additional_arg"]
        model = define_model(base_model, num_labels, label_embed_rank=label_embed_rank)
        logging.info(f"🏗️ Initialized L2R model with {num_labels} labels")

        # Move model to specified device
        model = model.to(device)
        logging.info(f"🖥️ Model moved to device: {device}")

        # Load custom weights
        safetensors_path = (
            os.path.join(model_path, "model.safetensors")
            if model_exists_locally
            else None
        )
        # The custom _save may write pytorch_model.bin (e.g. safe_serialization
        # off / tied embeddings) instead of model.safetensors, so check both --
        # locally and on the Hub -- rather than assuming safetensors and falling
        # through to an hf_hub_download of a local path (which raises HFValidationError).
        pytorch_bin_path = (
            os.path.join(model_path, "pytorch_model.bin")
            if model_exists_locally
            else None
        )
        try:
            if model_exists_locally and os.path.exists(safetensors_path):
                custom_state_dict = load_file(safetensors_path)
                source = "local path (safetensors)"
                files_dir = model_path  # Use model_path for local loading
            elif model_exists_locally and os.path.exists(pytorch_bin_path):
                custom_state_dict = torch.load(pytorch_bin_path, map_location="cpu")
                source = "local path (pytorch_model.bin)"
                files_dir = model_path
            else:
                api = HfApi(token=hf_token)
                try:
                    dl_path = api.hf_hub_download(
                        repo_id=model_path, filename="model.safetensors", token=hf_token
                    )
                    custom_state_dict = load_file(dl_path)
                except Exception:
                    dl_path = api.hf_hub_download(
                        repo_id=model_path, filename="pytorch_model.bin", token=hf_token
                    )
                    custom_state_dict = torch.load(dl_path, map_location="cpu")
                source = "Hub"
                files_dir = os.path.dirname(
                    dl_path
                )  # Use parent directory of the downloaded file for Hub loading
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

        # Load tokenizer from the true base model's Hub id (model_name),
        # falling back to hub_model_id only when model_name wasn't passed.
        # The tokenizer is never modified by LoRA fine-tuning or classify-head
        # training, and for custom-code tokenizers (e.g. Phi-3-small's
        # tiktoken-based Phi3SmallTokenizer), save_pretrained doesn't copy the
        # custom tokenization_*.py file into a local checkpoint dir -- and
        # hub_model_id itself is frequently a local dir here (e.g. a previous
        # stage's --skip_push_to_hub output), so reloading from it fails with
        # "does not appear to have a file named tokenization_*.py".
        tokenizer_source = model_name or hub_model_id
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source, token=hf_token, trust_remote_code=True
            )
            logging.info(f"🖌️ Loaded tokenizer from base model: {tokenizer_source}")
        except Exception as e:
            logging.error(f"❌ Failed to load tokenizer from {tokenizer_source}: {e}")
            raise

        # Move model to specified device
        model = model.to(device)
        logging.info(f"🖥️ Model moved to device: {device}")

        return model, tokenizer

    except Exception as e:
        logging.error(f"❌ Error in load_custom_task_model_and_tokenizer: {e}")
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


def prepare_training_components(
    custom_model,
    dataset,
    output_dir,
    task_hub_model_id,
    hf_token,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-4,
    warmup_steps=500,
    metric_for_best_model="f1_macro",
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
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        metric_for_best_model=metric_for_best_model,
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


def sample_df_with_full_label_coverage_old(
    df, frac=0.3, random_seed=42, max_attempts=10, verbose=True
):
    # Step 1: Parse label list and get full label set
    if "label_list" not in df.columns:
        df = df.copy()
        df["label_list"] = df["labels"].str.split(";")

    all_labels = set(chain.from_iterable(df["label_list"]))
    total_label_count = len(all_labels)

    if verbose:
        print(f"🔍 Total unique labels in dataset: {total_label_count}")

    for attempt in range(max_attempts):
        seed = random_seed + attempt
        sampled_df = pd.concat(
            [
                df[df["is_valid"]].sample(frac=frac, random_state=seed),
                df[~df["is_valid"]].sample(frac=frac, random_state=seed),
            ]
        )

        sample_labels = set(chain.from_iterable(sampled_df["labels"].str.split(";")))
        covered_label_count = len(sample_labels)

        if verbose:
            print(
                f"🧪 Attempt {attempt+1}: Sampled {len(sampled_df)} rows covering {covered_label_count} labels."
            )

        if sample_labels == all_labels:
            if verbose:
                print("✅ Full label coverage achieved.")
            return sampled_df

    # If still missing labels, fix greedily
    missing_labels = all_labels - sample_labels
    remaining_df = df.drop(index=sampled_df.index)

    if verbose:
        print(f"🛠️ Fixing missing labels: {len(missing_labels)} remaining...")

    # Greedy loop with tqdm
    added_rows = []
    with tqdm(total=len(missing_labels), desc="🔧 Filling label gaps") as pbar:
        while missing_labels:
            remaining_df["missing_label_overlap"] = remaining_df["label_list"].apply(
                lambda lbls: len(set(lbls) & missing_labels)
            )
            best_row = (
                remaining_df[remaining_df["missing_label_overlap"] > 0]
                .sort_values("missing_label_overlap", ascending=False)
                .head(1)
            )

            if best_row.empty:
                if verbose:
                    print("❌ Could not cover remaining missing labels. Stopping.")
                break

            added_rows.append(best_row)
            new_labels = set(best_row.iloc[0]["label_list"])
            covered_now = new_labels & missing_labels
            missing_labels -= covered_now
            pbar.update(len(covered_now))
            remaining_df = remaining_df.drop(index=best_row.index)

    if added_rows:
        fix_df = pd.concat(added_rows)
        sampled_df = pd.concat([sampled_df, fix_df], ignore_index=True)
        if verbose:
            print(f"✅ Added {len(added_rows)} rows to achieve full label coverage.")
            print(
                f"📊 Final total labels: {len(set(chain.from_iterable(sampled_df['label_list'])))}"
            )

    # Final count breakdown
    if verbose:
        n_valid = sampled_df["is_valid"].sum()
        n_not_valid = len(sampled_df) - n_valid
        print(
            f"✅ Final row count: {len(sampled_df)} (Valid: {n_valid}, Not-valid: {n_not_valid})"
        )

    return sampled_df


def sample_df_with_full_label_coverage(
    df,
    frac=0.3,
    random_seed=42,
    max_attempts=10,
    verbose=True,
    strict_label_coverage=True,
):
    # Step 1: Parse label list and get full label set
    if "label_list" not in df.columns:
        df = df.copy()
        df["label_list"] = df["labels"].str.split(";")

    all_labels = set(chain.from_iterable(df["label_list"]))
    total_label_count = len(all_labels)

    if verbose:
        logging.info(f"🔍 Total unique labels in dataset: {total_label_count}")

    for attempt in range(max_attempts):
        seed = random_seed + attempt
        sampled_df = pd.concat(
            [
                df[df["is_valid"]].sample(frac=frac, random_state=seed),
                df[~df["is_valid"]].sample(frac=frac, random_state=seed),
            ]
        )

        sample_labels = set(chain.from_iterable(sampled_df["labels"].str.split(";")))

        if verbose:
            logging.info(
                f"🧪 Attempt {attempt+1}: Sampled {len(sampled_df)} rows covering {len(sample_labels)} labels."
            )

        if sample_labels == all_labels:
            if verbose:
                n_valid = sampled_df["is_valid"].sum()
                n_not_valid = len(sampled_df) - n_valid
                logging.info("✅ Full label coverage achieved.")
                logging.info(
                    f"✅ Final row count: {len(sampled_df)} (Valid: {n_valid}, Not-valid: {n_not_valid})"
                )
            return sampled_df

    # If strict label coverage is not required, return as-is
    if not strict_label_coverage:
        if verbose:
            missing_labels = all_labels - sample_labels
            logging.info(
                f"⚠️ Skipping label coverage fix. {len(missing_labels)} labels are missing."
            )
            n_valid = sampled_df["is_valid"].sum()
            n_not_valid = len(sampled_df) - n_valid
            logging.info(
                f"✅ Final row count: {len(sampled_df)} (Valid: {n_valid}, Not-valid: {n_not_valid})"
            )
        return sampled_df

    # Greedy fix for missing labels
    missing_labels = all_labels - sample_labels
    remaining_df = df.drop(index=sampled_df.index)

    if verbose:
        logging.info(f"🛠️ Fixing missing labels: {len(missing_labels)} remaining...")

    added_rows = []
    with tqdm(total=len(missing_labels), desc="🔧 Filling label gaps") as pbar:
        while missing_labels:
            remaining_df["missing_label_overlap"] = remaining_df["label_list"].apply(
                lambda lbls: len(set(lbls) & missing_labels)
            )
            best_row = (
                remaining_df[remaining_df["missing_label_overlap"] > 0]
                .sort_values("missing_label_overlap", ascending=False)
                .head(1)
            )

            if best_row.empty:
                if verbose:
                    logging.info(
                        "❌ Could not cover remaining missing labels. Stopping."
                    )
                break

            added_rows.append(best_row)
            new_labels = set(best_row["label_list"].values[0])
            covered_now = new_labels & missing_labels
            missing_labels -= covered_now
            pbar.update(len(covered_now))
            remaining_df = remaining_df.drop(index=best_row.index)

    if added_rows:
        fix_df = pd.concat(added_rows)
        sampled_df = pd.concat([sampled_df, fix_df], ignore_index=True)
        if verbose:
            logging.info(
                f"✅ Added {len(added_rows)} rows to achieve full label coverage."
            )
            logging.info(
                f"📊 Final total labels: {len(set(chain.from_iterable(sampled_df['label_list'])))}"
            )

    # Final count breakdown
    if verbose:
        n_valid = sampled_df["is_valid"].sum()
        n_not_valid = len(sampled_df) - n_valid
        logging.info(
            f"✅ Final row count: {len(sampled_df)} (Valid: {n_valid}, Not-valid: {n_not_valid})"
        )

    return sampled_df


# Function to truncate and wrap
def wrap_and_truncate(val, width=100):
    val = str(val)[:width] + "..." if len(str(val)) > width else str(val)
    return "\n".join(textwrap.wrap(val, width=width))


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


def is_valid_hf_model(model_id):
    """Check if a model ID exists on Hugging Face Hub"""
    try:
        model_info(model_id)
        logging.info(f"✅ Validated Hugging Face model: {model_id}")
        return True
    except (RepositoryNotFoundError, Exception) as e:
        logging.info(f"❌ Invalid Hugging Face model: {model_id} - {str(e)}")
        return False


# =============================================================================
# Main Function
# =============================================================================
def main(
    output_dir: str = "task2_output",
    source_url: str = "XURLs.MIMIC4",
    data: str = "mimic4_icd10_full",
    logfile: str = "classification_log",
    base_dir: str = None,
    hub_model_id: str = "deb101/mistral-7b-mimic4",  # Task 1 Domain Adaptation model
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    max_length: int = 512,
    do_fresh_training: bool = False,
    load_from_checkpoint: bool = False,
    task: str = "multilabel-classify",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    metric_for_best_model: str = "f1_macro",
    learning_rate: float = 2e-4,
    final_lr_scheduling: float = 1e-6,
    warmup_steps: int = 500,
    skip_push_to_hub: bool = False,
    rare_threshold: int = 50,
    label_coverage_frac: float = 0.001,
    label_space: str = "label",
    candidate_budget: int = 256,
    label_embed_rank: int = None,
):
    """ """
    # -----------------------------------------------------------------------------
    # Data and File Setup
    # -----------------------------------------------------------------------------
    # Setup logger
    logfile_path = setup_logging(**locals())
    log_section("INITIALIZING TRAINING ENVIRONMENT", "=")

    logging.info("🚀 Setting up data paths and environment variables...")

    source = untar_xxx(eval(source_url))
    base_dir = base_dir + "/" + source_url.split(".")[1]
    output_dir = str(Path(base_dir) / output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"📂 Using output directory: {output_dir}")

    # print arguments
    print_args(**locals())

    # record git repo stats
    logging.info(f"\n🚀 Quick Git Info: {get_compact_git_info()}")

    log_section("LOADING DATASETS", "+")
    logging.info("📊 Loading main datasets....")

    # Load your DataFrame
    df = get_data(source, data)

    # quick iteration <remove later>
    # # Select and copy the row to avoid modifying original DataFrame
    # row = df.iloc[3:10].copy()
    # row['text'] = row['text'].apply(lambda x: wrap_and_truncate(x, width=40))
    # row['labels'] = row['labels'].apply(lambda x: wrap_and_truncate(x, width=40))
    # print(tabulate(row, headers='keys', tablefmt='psql', showindex=False))

    df = sample_df_with_full_label_coverage(
        df,
        frac=label_coverage_frac,
        max_attempts=10,
        random_seed=random_seed,
        strict_label_coverage=True,
    )
    # quick iteration <remove later>

    # log_with_box("****Multi-Label Classification Started****")
    log_section("STARTING MULTI_LABEL CLASSIFICATION MODEL TRAINING", "*")

    # Hardcode your HF token (or load from environment variable)
    # hf_token = "your_hf_token_here"  # Replace or use os.environ["HF_TOKEN"]
    hf_token = os.environ["HF_TOKEN"]
    logging.info("🔐 Loaded authentication token from environment")

    # Create task specific hub model id
    task_hub_model_id = f"{hub_model_id}-{task}"
    logging.info(f"🏷️ Hub Model ID for this Classification task: {task_hub_model_id}")

    log_section("MODEL EXISTENCE CHECK", "-")

    # Check if model exists locally or on Hub (unless do_fresh_training is True)
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
                log_section("LOADING FROM CHECKPOINT", "+")
                logging.info("📥 Loading existing model from checkpoint...")
                # Load the custom task specific model and the tokenizer
                custom_model, tokenizer = load_custom_task_model_and_tokenizer(
                    output_dir,
                    hub_model_id,
                    task_hub_model_id,
                    model_exists_locally,
                    do_fresh_training,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    hf_token=hf_token,
                    skip_push_to_hub=skip_push_to_hub,
                    model_name=model_name,
                )
                logging.info(
                    f"✅ Successfully loaded from checkpoint, will now continue training for {num_train_epochs} epochs."
                )
                custom_model.train()

                # Preprocess data
                log_section("PREPARING EVALUATION DATA", "+")
                logging.info("🔄 Loading and preprocessing evaluation data...")
                dataset, icd_codes, mlb, rare_labels, not_rare_labels, zero_labels = preprocess_data(
                    df, tokenizer, base_dir=base_dir, max_length=max_length, rare_threshold=rare_threshold,
                    label_space=label_space, candidate_budget=candidate_budget,
                )
                logging.info(f"🏷️ Number of unique ICD-10 codes: {len(icd_codes)}")
            else:
                log_section("LOADING BASE MODEL", "+")
                logging.info("📥 Loading pretrained model and tokenizer...")
                # Load Task 1 (Domain Adapted) model and tokenizer
                # base_model, tokenizer = load_base_model_and_tokenizer(
                #     model_name, hub_model_id, task_hub_model_id, hf_token
                # )
                base_model, tokenizer = load_base_model_and_tokenizer(
                    hub_model_id, model_name=model_name, hf_token=hf_token
                )

                log_section("DATA PREPROCESSING", "+")
                logging.info("🔄 Loading and preprocessing training data...")
                # Preprocess data
                dataset, icd_codes, mlb, rare_labels, not_rare_labels, zero_labels = preprocess_data(
                    df, tokenizer, base_dir=base_dir, max_length=max_length, rare_threshold=rare_threshold,
                    label_space=label_space, candidate_budget=candidate_budget,
                )
                num_labels = len(icd_codes)
                logging.info(f"Number of unique ICD-10 codes: {num_labels}")

                log_section("MODEL INITIALIZATION", "+")
                logging.info(
                    "🧠 Initializing custom L2R model for outputting per-token relevance scores per ICD-10 codes."
                )
                # Define custom model
                custom_model = define_model(base_model, num_labels, label_embed_rank=label_embed_rank)

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
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                    learning_rate=learning_rate,
                    warmup_steps=warmup_steps,
                    metric_for_best_model=metric_for_best_model,
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
                rare_labels=rare_labels,
                not_rare_labels=not_rare_labels,
                zero_labels=zero_labels,
                final_lr_scheduling=final_lr_scheduling,
                label_space=label_space,
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
        finally:
            sys.exit(0)
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

    log_section("STARTING MODEL EVALUATION", "*")
    logging.info("📊 Running comprehensive model evaluation...")

    # Load the custom task specific model and the tokenizer
    custom_model, tokenizer = load_custom_task_model_and_tokenizer(
        output_dir,
        hub_model_id,
        task_hub_model_id,
        model_exists_locally,
        do_fresh_training,
        device="cuda" if torch.cuda.is_available() else "cpu",
        hf_token=hf_token,
        skip_push_to_hub=skip_push_to_hub,
    )

    # Preprocess data
    log_section("PREPARING EVALUATION DATA", "+")
    logging.info("🔄 Loading and preprocessing evaluation data...")
    dataset, icd_codes, mlb, rare_labels, not_rare_labels, zero_labels = preprocess_data(
        df, tokenizer, base_dir=base_dir, max_length=max_length, rare_threshold=rare_threshold,
        label_space=label_space, candidate_budget=candidate_budget,
    )
    logging.info(f"🏷️ Number of unique ICD-10 codes: {len(icd_codes)}")

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
            per_device_eval_batch_size=per_device_eval_batch_size,
            push_to_hub=not skip_push_to_hub,
        )
    )

    # Evaluate with train_model
    log_section("RUNNING EVALUATION", "+")
    logging.info("📏 Evaluating model performance on test dataset...")
    logging.info("📊 Using k values of 1, 5, 8, 15, and 25 for evaluation metrics...")
    trainer = train_model(
        custom_model,
        training_args,
        train_dataset,
        eval_dataset,
        tokenizer,
        k_values=[1, 5, 8, 15, 25],
        rare_labels=rare_labels,
        not_rare_labels=not_rare_labels,
        zero_labels=zero_labels,
        eval=True,
        label_space=label_space,
    )

    log_section("EVALUATION COMPLETE", "=")
    logging.info("🏆 Model evaluation completed successfully.")

    sys.exit(0)


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    # Convert the Namespace to a dictionary and pass to main via dictionary unpacking.
    main(**vars(args))
