import os
import yaml
from accelerate.state import AcceleratorState
from pprint import pprint

# Function to prompt user for training type
def get_training_type():
    print("Select training type:")
    print("1. Non-distributed training (single GPU/CPU)")
    print("2. Distributed training (multi-GPU)")
    print("3. Distributed training with DeepSpeed")
    while True:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        if choice in ["1", "2", "3"]:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")

# Build the base config dictionary
config = {
    "compute_environment": "LOCAL_MACHINE",
    "mixed_precision": "fp16",               # Mixed precision
    "use_cpu": False,                        # Use GPU if available
    "dynamo_backend": "NO",                  # No Torch Dynamo
    "main_training_function": "main",        # Entry point function
    "num_machines": 1,
    "machine_rank": 0,
}

# Get user input for training type
training_type = get_training_type()

# Configure based on user choice
if training_type == "1":
    # Non-distributed training
    config["distributed_type"] = "NO"
    config["num_processes"] = 1
    config["gpu_ids"] = "all"
elif training_type == "2":
    # Distributed training (multi-GPU)
    config["distributed_type"] = "MULTI_GPU"
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        config["num_processes"] = max(1, num_gpus)  # Use all available GPUs, at least 1
        config["gpu_ids"] = ",".join(str(i) for i in range(num_gpus)) if num_gpus > 0 else "0"
    except ImportError:
        print("⚠️ PyTorch not installed. Defaulting to single process.")
        config["num_processes"] = 1
        config["gpu_ids"] = "0"
elif training_type == "3":
    # Distributed training with DeepSpeed
    config["distributed_type"] = "DEEPSPEED"
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        config["num_processes"] = max(1, num_gpus)  # Use all available GPUs, at least 1
        config["gpu_ids"] = ",".join(str(i) for i in range(num_gpus)) if num_gpus > 0 else "0"
    except ImportError:
        print("⚠️ PyTorch not installed. Defaulting to single process.")
        config["num_processes"] = 1
        config["gpu_ids"] = "0"

# Check GPU availability and adjust if necessary
try:
    import torch
    if not torch.cuda.is_available() and not config["use_cpu"]:
        print("⚠️ No GPU available. Switching to CPU.")
        config["use_cpu"] = True
        config["gpu_ids"] = None
except ImportError:
    print("⚠️ PyTorch not installed. Assuming CPU-only setup.")
    config["use_cpu"] = True
    config["gpu_ids"] = None

# Define the default config path
default_config_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "accelerate", "default_config.yaml")

# Ensure the directory exists
os.makedirs(os.path.dirname(default_config_path), exist_ok=True)

# Delete the old configuration file if it exists
if os.path.exists(default_config_path):
    try:
        os.remove(default_config_path)
        print(f"🗑️ Deleted old configuration file: {default_config_path}")
    except Exception as e:
        print(f"❌ Failed to delete old config: {e}")
        exit(1)

# Save the full config dictionary to the YAML file
try:
    with open(default_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"✅ Accelerate config saved to: {default_config_path}\n")
except Exception as e:
    print(f"❌ Failed to save config: {e}")
    exit(1)

# Show final configuration
print("📄 Final Configuration:")
pprint(config)

# Optional: Reset the global state
AcceleratorState._reset_state()
print("🔄 Accelerator state reset successfully.")