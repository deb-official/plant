import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from huggingface_hub import hf_hub_download
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import your custom LTRModel and LTRConfig
# Assuming LTRModel and LTRConfig are defined in a module named `ltr_model.py`
from xcube.l2r.models.core import LTRModel, LTRConfig


def load_ltr_model(repo_id, hf_token=None):
    """
    Load the LTRModel from a Hugging Face repository.

    Args:
        repo_id (str): Hugging Face repository ID (e.g., "deb101/mistral-7b-instruct-v0.3-mimic4-adapt-l2r").
        hf_token (str, optional): Hugging Face access token for private repositories.

    Returns:
        model: Loaded LTRModel instance.
        tokenizer: Tokenizer for the ground model.
    """
    try:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        # Load model
        logging.info(f"Loading LTRModel from {repo_id}...")
        model = LTRModel.from_pretrained(
            pretrained_model_name_or_path=repo_id,
            token=hf_token,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16,
        )
        model.eval()  # Set model to evaluation mode
        logging.info("Model loaded successfully.")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            token=hf_token,
        )
        logging.info("Tokenizer loaded successfully.")

        return model, tokenizer

    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        raise


def prepare_inputs(texts, tokenizer, max_length=128):
    """
    Tokenize input texts and prepare input tensors with bfloat16 dtype.

    Args:
        texts (list[str]): List of input texts to tokenize.
        tokenizer: Tokenizer instance.
        max_length (int): Maximum sequence length.

    Returns:
        dict: Dictionary containing input_ids and attention_mask.
    """
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "input_ids": encodings["input_ids"].to(device=device, dtype=torch.long),
        "attention_mask": encodings["attention_mask"].to(
            device=device, dtype=torch.long
        ),
    }


def main():
    # Hugging Face repository and optional token
    repo_id = "deb101/mistral-7b-instruct-v0.3-mimic4-adapt-l2r"
    hf_token = (
        None  # Replace with your friend's Hugging Face token if the repo is private
    )

    # Load model and tokenizer
    model, tokenizer = load_ltr_model(repo_id, hf_token)

    # Example input texts (replace with your actual input data)
    texts = [
        "Patient with chest pain and shortness of breath.",
        "History of diabetes and hypertension.",
    ]

    # Prepare inputs
    inputs = prepare_inputs(texts, tokenizer, max_length=128)

    # Debug dtypes and devices
    logging.info(
        f"input_ids dtype: {inputs['input_ids'].dtype}, device: {inputs['input_ids'].device}"
    )
    logging.info(
        f"attention_mask dtype: {inputs['attention_mask'].dtype}, device: {inputs['attention_mask'].device}"
    )
    logging.info(
        f"Model parameters dtype: {next(model.parameters()).dtype}, device: {next(model.parameters()).device}"
    )
    logging.info(
        f"Label embeddings dtype: {model.label_embeddings.weight.dtype}, device: {model.label_embeddings.weight.device}"
    )
    logging.info(
        f"Ground Model parameters dtype: {next(model.ground_model.parameters()).dtype}, device: {next(model.parameters()).device}"
    )
    if model.lookup_table is not None:
        logging.info(
            f"Lookup table dtype: {model.lookup_table.dtype}, device: {model.lookup_table.device}"
        )

    # Move model to the correct device (already handled in from_pretrained, but confirm)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Perform inference
    try:
        with torch.no_grad(), torch.amp.autocast(device_type=device):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

        # Extract outputs
        logits = outputs[
            "logits"
        ]  # Relevance scores: (batch_size, max_len, num_labels)
        hidden_states = outputs["hidden_states"]  # Hidden states from the model
        loss = outputs.get("loss", None)  # Loss (if labels are provided)

        # Print results
        logging.info(f"Relevance scores shape: {logits.shape}")
        logging.info(f"Hidden states shape: {hidden_states.shape}")
        if loss is not None:
            logging.info(f"Loss: {loss.item()}")

        # Example: Print relevance scores for the first input
        print("Relevance scores for first input:", logits[0].cpu().numpy())

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()
