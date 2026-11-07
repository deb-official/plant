# =============================================================================
# Imports
# =============================================================================
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.data.core import *
from xcube.l2r.all import *
from transformers import AutoTokenizer
from huggingface_hub import login, HfApi, hf_hub_download
from fastcore.foundation import L
torch.serialization.add_safe_globals([L])
from tqdm import tqdm
import shutil

# extra imports <remove later>
import warnings; warnings.filterwarnings(action='ignore')
# end extra imports

# =============================================================================
# Constants
# =============================================================================
torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80
# Initialize generated_files list with tuples of (file_path, description)
generated_files = []

# =============================================================================
# Warnings and Configuration
# =============================================================================
warnings.filterwarnings(action='ignore')
torch.backends.cudnn.benchmark = True

# =============================================================================
# Helper Functions
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Script for processing MIMIC4 data")
    
    # Define arguments with defaults and help text
    parser.add_argument(
        "--source_url",
        type=str,
        default="XURLs.MIMIC4_L2R",
        help="Source url (default: %(default)s)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mimic4_icd10_full",
        help="Filename of the raw data (default: %(default)s)"
    )
    parser.add_argument(
        "--label_delim",
        type=str,
        default=";",
        help="Delimiter separating labels (default: %(default)s)"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="..",
        help="Root dir for saving models (default: %(default)s)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers (default: %(default)s)"
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=16,
        help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--chnk_sz",
        type=int,
        default=200,
        help="Chunk size (default: %(default)s)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output path for the finetuned LLM model."
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="l2rboot",
        help="Filename to write logs to"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Path to the base directory that will contain intermediate results and models."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="your_username/mistral-7b-medical-lora",
        help="Hugging Face hub model id (replace with your username/model name)"
    )
    parser.add_argument(
    "--max_length",
    type=int,
    default=512,
    help="Maximum sequence length. Adjust as needed (e.g., 1024 if your summaries are longer and 48GB VRAM allows)."
    )
    parser.add_argument(
        "--label_space",
        type=str,
        default="label",
        choices=["label", "cluster"],
        help="'label' (default) runs the MIG bootstrap once over the full label vocabulary, as usual. "
        "'cluster' runs it once per label cluster instead (Phase 4 of the full-scale candidate-generator "
        "plan, see README_AMAZON3M.md) -- requires cluster_assignments.json/cluster_to_labels.json "
        "(from scripts/build_label_clusters.py) to already exist under --base_dir. Each cluster's dense "
        "token-label tensors stay bounded by that cluster's (small) label count instead of the full "
        "vocabulary, and outputs are written per-cluster (fname suffixed with _cluster{id}) instead of "
        "one combined file.",
    )
    parser.add_argument(
        "--ckpt_hub_repo",
        type=str,
        default=None,
        help="Optional HF Hub dataset repo (e.g. 'youruser/wiki10-l2rboot-ckpt'). If "
        "set, each expensive intermediate (joint pmf from the epoch loop, info-theoretic "
        "metrics, Box-Cox output, the two rank tables) is pushed there right after it's "
        "computed, and pulled from there before recomputing if missing locally -- so a "
        "crash (or losing the box entirely) doesn't mean redoing the ~7-8hr epoch loop "
        "from scratch. Local-disk checkpointing/resume happens either way; this just adds "
        "Hub as a second copy. Best-effort: a Hub failure is logged, never aborts the run.",
    )

    return parser.parse_args()


def print_args(**kwargs):
    """
    Logs and prints the provided keyword arguments in a formatted manner.
    
    Args:
        **kwargs: Arbitrary keyword arguments representing command-line parameters.
    """
    # Create a formatted string of key-value pairs.
    formatted_args = "\n".join(f"{key}: {value}" for key, value in kwargs.items())
    
    # Log and print the formatted arguments.
    logging.info("Command-line Arguments:")
    logging.info("\n" + formatted_args)


# Configure logging
def setup_logging(**kwargs):
    """
    Configure logging for the application.

    Sets up both console and file logging based on the quiet flag in args.
    Logs include detailed information such as function name, line number,
    and timestamp. Log files are saved in a 'logs' directory within base_dir
    with a timestamp in their filename.
    """

    # Access the 'base_dir' and 'logfile' arguments from kwargs.
    quiet = kwargs.get('quiet')
    logfile = kwargs.get('logfile')
    base_dir = kwargs.get('base_dir')

    # Configure the root logger based on the quiet flag
    if quiet:
        logging.basicConfig(level=logging.CRITICAL)  # Suppresses all logging below CRITICAL
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d:%(funcName)s]'
        )

    # Set up file logging globally
    log_base_dir = Path(base_dir) / 'logs'
    log_base_dir.mkdir(exist_ok=True, parents=True)

    # Format the current date and time for the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = (
        f'{logfile}_{current_time}.log'
        if logfile
        else f'default_logfile_{current_time}.log'
    )
    log_filepath = log_base_dir / log_filename

    # Create file handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)  # Set logging level for the file handler

    # Create formatter and add it to the file handler
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d:%(funcName)s]'
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


def get_info(source,
             data,
             hub_model_id,
             base_dir,
             hf_token,
             bs=8,
             chnk_sz=200,
             workers=None,
             mlb_classes_override=None,  # restrict to this label subset instead of the full vocabulary (per-cluster Stage 3, see main())
             ):
    workers = ifnone(workers, min(8, num_cpus()))
    data = join_path_file(data, source, ext='.csv')
    df = pd.read_csv(data,
                 header=0,
                 usecols=['text', 'labels'],
                 dtype={'text': str, 'labels': str})
    df[['text', 'labels']] = df[['text', 'labels']].astype(str)
    logging.info(f"The size of the original dataframe is {len(df)}")

    # # sampling for quick iterations: remove later 
    # bs = 8
    # cut = len(df) - len(df)%bs
    # df = df[:cut]
    # len(df)

    # _arr = np.arange(0, len(df), bs)
    # # mask = (_arr > 4000) & (_arr < 5000)
    # mask = (_arr > 500) & (_arr < 1000)
    # _n = np.random.choice(_arr[mask], 1)
    # df = df.sample(n=_n, random_state=89, ignore_index=True)
    # logging.info(f"For quick iteration sampled a dataframe of size {len(df)}")
    # # sampling for quick iterations: remove later 

    # Duplicate the last row # this is a hack <fix later>
    last_rows = df.iloc[-14:]  # Get the last row as a DataFrame
    df = pd.concat([df, last_rows], ignore_index=True)
    # this is a hack <fix later>

    logging.info(f"Loading the Tokenizer from {hub_model_id}")
    api = HfApi()
    repo_info = api.repo_info(repo_id=hub_model_id, token=hf_token)
    last_modified = repo_info.last_modified  # This is a datetime object
    if last_modified:
        last_modified_str = last_modified.strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"Model {hub_model_id} was last saved on: {last_modified_str}")
    tokenizer = AutoTokenizer.from_pretrained(
        hub_model_id, token=hf_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mlb_classes_override is not None:
        mlb_classes = mlb_classes_override
        logging.info(f"Using a {len(mlb_classes)}-label override instead of the full vocabulary (per-cluster Stage 3 run).")
    else:
        mlb_classes = load_mlb_classes(mlb_dir=base_dir)
        if mlb_classes: logging.info(f"Loading of mlb classes from {base_dir} was successfull!")

    logging.info(f"Reading the data from {source/data} into dataframe!")
    logging.info(f"The length of the dataframe is {len(df)}, an example text (first 500 words) is \n -----\n{df.iloc[0].text[:500]} \n------\n and the corresponding labels are \n------\n {df.iloc[0].labels}\n---\n")
    info = MutualInfoGain(df, tokenizer, mlb_classes, bs=bs, chnk_sz=chnk_sz, lbs_desc=None) # provide lbs_desc if you have it
    return info


def test_nanegs(**kwargs):
    for k, v in kwargs.items():
        has_nans = v.isnan().all() # check for nans
        has_negs = not torch.where(v>=0, True, False).all()
        # if has_nans: raise Exception(f"{namestr(o, locals())[0]} has nans")
        # if has_negs: raise Exception(f"{namestr(o, locals())[0]} has negs")
        if has_nans: raise Exception(f"{k} has nans")
        if has_negs: raise Exception(f"{k} has negs")


def stat_transform_old(mut_infos, save_dir, fname):
    logging.info(f"{mut_infos.shape=}")
    orig_shape = mut_infos.shape
    mut_infos = mut_infos.cpu().numpy().reshape(-1)
    logging.info(f"{mut_infos.min()=}, {mut_infos.max()=}, {mut_infos.mean()=}")
    logging.info(f"{skew(mut_infos)=}")
    logging.info("The mutual-info values are incredibly skewed. So we need to apply some transformation. Sometimes `mut_infos` might contain negs, we need to convert those to eps.")
    # np.where(mut_infos<0, 1, 0).sum() # or, better yet
    where_negs = mut_infos < 0
    logging.info(f"{np.sum(where_negs)=}")
    logging.info("Converting the negs into eps..")
    eps = np.float32(1e-20)
    mut_infos[where_negs] = eps
    test_eq(np.sum(mut_infos<0), 0)
    logging.info(f"{ np.min(mut_infos) = }, { np.max(mut_infos) = }, { np.mean(mut_infos) = }");
    hist, bins, _ = plt.hist(mut_infos, bins=50)
    f = join_path_file(fname+'_mut_infos_hist', save_dir, ext='.png'); plt.savefig(f); generated_files.append(f)
    logging.info("Applying box-cox transformation...")
    bcx_mut_infos, *_ = boxcox(mut_infos+eps)
    logging.info(f"{np.min(bcx_mut_infos) = }, {np.max(bcx_mut_infos)=}, {np.mean(bcx_mut_infos) = }, {np.median(bcx_mut_infos) = }")
    logging.info(f"{np.isnan(bcx_mut_infos).sum() = }, {np.isinf(bcx_mut_infos).sum() = }, {np.isneginf(bcx_mut_infos).sum() = }")
    logging.info(f"{skew(bcx_mut_infos) = }")
    hist, bins, _ = plt.hist(bcx_mut_infos, bins=50)
    f=join_path_file(fname+'_bcx_mut_infos_hist', save_dir, ext='.png'); plt.savefig(f); generated_files.append(f)
    return bcx_mut_infos.reshape(*orig_shape)


def boxcox_gpu(x, lambda_val=None, eps=1e-6, sample_size=100000, device="cuda", per_column=False):
    """
    Apply Box-Cox transformation on GPU using PyTorch, with optional lambda estimation.
    
    Args:
        x (torch.Tensor or np.ndarray): Input tensor or NumPy array (will be moved to GPU)
        lambda_val (float, optional): Box-Cox parameter; if None, estimated from data
        eps (float): Small value to ensure positivity (default: 1e-6)
        sample_size (int): Size of random subsample for lambda estimation (default: 50000)
        device (str or torch.device): Target device (default: "cuda")
        per_column (bool): If True, estimate lambda per column; if False, global lambda (default: False)
    
    Returns:
        torch.Tensor: Transformed tensor on the specified device
    """
    # Ensure device is a torch.device object
    device = torch.device(device) if isinstance(device, str) else device
    
    # Convert NumPy array to PyTorch tensor if necessary, then move to device
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device) if x.device != device else x
    
    # Ensure input is strictly positive by shifting if necessary
    min_val = x.min()
    if min_val <= 0:
        x_positive = x - min_val + eps  # Shift to make all values > 0
    else:
        x_positive = x + eps
    
    # Estimate lambda if not provided
    if lambda_val is None:
        if per_column:
            # Estimate lambda for each column
            n_rows, n_cols = x_positive.shape
            sample_size = min(sample_size, n_rows)
            if sample_size < 2:
                raise ValueError("Sample size must be at least 2 for lambda estimation")
            lambda_vals = torch.zeros(n_cols, device=device, dtype=x.dtype)
            for col in tqdm(range(n_cols), desc="Estimating per-column lambda"):
                col_data = x_positive[:, col]
                indices = torch.randperm(n_rows, device=device)[:sample_size]
                subsample = col_data[indices].cpu().numpy()
                try:
                    _, lambda_val = boxcox(subsample)
                    lambda_vals[col] = lambda_val
                except ValueError:
                    lambda_vals[col] = 0.0  # Fallback to log if estimation fails
        else:
            # Global lambda estimation
            flat_x = x_positive.flatten()
            n_elements = flat_x.numel()
            sample_size = min(sample_size, n_elements)
            if sample_size < 2:
                raise ValueError("Sample size must be at least 2 for lambda estimation")
            indices = torch.randperm(n_elements, device=device)[:sample_size]
            subsample = flat_x[indices].cpu().numpy()
            _, lambda_val = boxcox(subsample)
            lambda_vals = torch.tensor(lambda_val, device=device, dtype=x.dtype)
    else:
        lambda_vals = torch.tensor(lambda_val, device=device, dtype=x.dtype) if not isinstance(lambda_val, torch.Tensor) else lambda_val.to(device)
    
    # Apply Box-Cox transformation on GPU
    if torch.is_tensor(lambda_vals) and lambda_vals.dim() > 0:  # Per-column lambda
        out = torch.zeros_like(x_positive)
        for col in tqdm(range(x_positive.shape[1]), desc="Applying per-column Box-Cox"):
            if lambda_vals[col] == 0:
                out[:, col] = torch.log(x_positive[:, col])
            else:
                out[:, col] = (x_positive[:, col].pow(lambda_vals[col]) - 1) / lambda_vals[col]
        return out
    else:  # Global lambda
        if lambda_vals == 0:
            return torch.log(x_positive)
        else:
            return (x_positive.pow(lambda_vals) - 1) / lambda_vals


def stat_transform(mut_infos, save_dir, fname):
    """
    Apply statistical transformations to mutual information values.
    
    This function performs Box-Cox transformation on mutual information values
    to correct for skewness. It also generates histograms before and after transformation.
    
    Args:
        mut_infos (Tensor): Tensor containing mutual information values
        save_dir (Path): Directory where to save histogram plots
        fname (str): Base filename for saved plots
        
    Returns:
        np.ndarray: Transformed mutual information values with original shape
    """
    # Initialize generated_files list if not already defined
    if 'generated_files' not in locals():
        generated_files = []
        
    # -----------------------------------------------------------------------------
    # Initial Analysis and Preprocessing
    # -----------------------------------------------------------------------------
    logging.info(f"Mutual Info Shape: {mut_infos.shape}")
    orig_shape = mut_infos.shape

    # Keep as PyTorch tensor and maintain original shape
    logging.info(
        f"Raw Mutual Info Stats: Min={torch.min(mut_infos):.8f}, "
        f"Max={torch.max(mut_infos):.8f}, "
        f"Mean={torch.mean(mut_infos):.8f}"
    )

    # For skewness in PyTorch without flattening
    # Skewness = E[(X-μ)³]/σ³
    mean = torch.mean(mut_infos)
    std = torch.std(mut_infos)
    skewness = torch.mean(((mut_infos - mean) / std) ** 3) if std > 0 else torch.tensor(0.0)
    logging.info(f"Distribution Skewness: {skewness:.6f}")

    logging.info(
        "The mutual-info values are incredibly skewed. So we need to apply some transformation. "
        "Sometimes `mut_infos` might contain negatives, we need to convert those to epsilon."
    )

    # Handle negative values while preserving shape
    where_negs = mut_infos < 0
    neg_count = torch.sum(where_negs).item()
    logging.info(f"Negative Values Count: {neg_count}")

    logging.info("Converting the negative values to epsilon...")
    eps = torch.tensor(1e-20, dtype=mut_infos.dtype, device=mut_infos.device)
    mut_infos = torch.where(mut_infos < 0, eps, mut_infos)
    assert torch.sum(mut_infos < 0).item() == 0

    logging.info(
        f"After Fixing Negatives: Min={torch.min(mut_infos):.8e}, "
        f"Max={torch.max(mut_infos):.8e}, "
        f"Mean={torch.mean(mut_infos):.8e}"
    )
    
    # -----------------------------------------------------------------------------
    # Original Distribution Histogram
    # -----------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Calculate histogram with PyTorch while preserving dimensionality
    hist_values, bin_edges = torch.histogram(mut_infos, bins=50)

    # Plot the histogram using bar plot
    plt.bar(
        bin_edges[:-1].cpu().numpy(), 
        hist_values.cpu().numpy(), 
        width=(bin_edges[1] - bin_edges[0]).item(),
        align='edge'
    )
    plt.title('Mutual Information Distribution (Original)')
    plt.xlabel('Mutual Information Value')
    plt.ylabel('Frequency')

    # Save original histogram
    orig_hist_file = join_path_file(fname + '_mut_infos_hist', save_dir, ext='.png')
    logging.info(f"Saving original distribution histogram to {orig_hist_file}...")
    plt.savefig(orig_hist_file)
    logging.info(f"Successfully saved original distribution histogram to {orig_hist_file}")
    generated_files.append((orig_hist_file, "Histogram of original mutual information distribution"))
    plt.close()

    # -----------------------------------------------------------------------------
    # Box-Cox Transformation
    # -----------------------------------------------------------------------------
    logging.info("Applying Box-Cox transformation...")
    # Timing setup
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    # scipy.stats.boxcox's own lambda-MLE search evaluates the log-likelihood
    # over the FULL array for every candidate lambda tried during its internal
    # 1D optimization -- at Wiki10-31K full scale (toksize=128256 x
    # lblsize=30938 ~= 4B elements), each evaluation needs several
    # full-array-sized temporaries, and enough stack up simultaneously to
    # exceed even this box's 525GB system RAM -- killed by the Linux
    # OOM-killer (SIGKILL, not a catchable Python exception) right at this
    # step, confirmed directly (Stage 3 reached 100% of all 155 label chunks,
    # completed the joint_pmf/compute() fix above cleanly, then died here).
    # Use boxcox_gpu instead (already written above for exactly this problem,
    # just never wired in): estimate lambda from a cheap 100K-element
    # subsample via scipy (as before, just on a subsample, not all ~4B
    # elements), then apply the transform formula directly in a single
    # elementwise pass over the full tensor -- no repeated full-array
    # evaluations. per_column=False, not the per_column=True this call used
    # before being commented out: per-column mode had an unrelated bug (loop
    # variable `colrome` vs. the referenced `col`, fixed above) which was
    # likely why this got reverted to the naive scipy call rather than fixed
    # -- per-label lambda estimation also isn't needed here; a single global
    # lambda for the whole distribution is standard Box-Cox usage.
    # device=mut_infos.device (not hardcoded "cuda"): mut_infos is CPU-resident
    # by this point (see the joint_pmf/compute() CPU fix above), and there is
    # no need to round-trip it through the GPU for a plain elementwise pass.
    bcx_mut_infos = boxcox_gpu(mut_infos, device=mut_infos.device, per_column=False, sample_size=100000)
    bcx_mut_infos_np = bcx_mut_infos.cpu().numpy().reshape(*orig_shape)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    logging.info(f"boxcox_gpu took {elapsed_time_ms / 60000:.2f} minutes")

    # Check skewness after boxcox transformation
    skewness = skew(bcx_mut_infos_np.flatten(), bias=True)
    logging.info(f"Skewness after boxcox transformation: {skewness:.4f}")
    
    # Log statistics about transformed values
    logging.info(
        f"BCX Mutual Info Stats: Min={np.min(bcx_mut_infos_np):.6f}, "
        f"Max={np.max(bcx_mut_infos_np):.6f}, "
        f"Mean={np.mean(bcx_mut_infos_np):.6f}, "
        f"Median={np.median(bcx_mut_infos_np):.6f}"
    )

    logging.info(
        f"BCX Mutual Info Issues: NaN Count={np.isnan(bcx_mut_infos_np).sum()}, "
        f"Inf Count={np.isinf(bcx_mut_infos_np).sum()}, "
        f"Neg Inf Count={np.isneginf(bcx_mut_infos_np).sum()}"
    )
    
    # -----------------------------------------------------------------------------
    # Transformed Distribution Histogram
    # -----------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Calculate histogram with PyTorch while preserving dimensionality
    hist_values, bin_edges = np.histogram(bcx_mut_infos_np, bins=50)

    # Plot the histogram using bar plot
    plt.bar(
        bin_edges[:-1], 
        hist_values, 
        width=bin_edges[1] - bin_edges[0],
        align='edge'
    )
    plt.title('Mutual Information Distribution (Box-Cox Transformed)')
    plt.xlabel('Transformed Mutual Information Value')
    plt.ylabel('Frequency')

    # Save transformed histogram
    transformed_hist_file = join_path_file(fname + '_bcx_mut_infos_hist', save_dir, ext='.png')
    logging.info(f"Saving transformed distribution histogram to {transformed_hist_file}...")
    plt.savefig(transformed_hist_file)
    logging.info(f"Successfully saved transformed distribution histogram to {transformed_hist_file}")
    generated_files.append((transformed_hist_file, "Histogram of Box-Cox transformed mutual information distribution"))
    plt.close()
    
    # Return transformed values with original shape
    # return bcx_mut_infos_np.reshape(*orig_shape)
    return bcx_mut_infos_np


def push_ckpt_to_hub(local_path, ckpt_hub_repo, hf_token):
    """Best-effort upload of a local checkpoint file to ckpt_hub_repo (a HF
    Hub dataset repo), under its own basename. Never raises: a Hub push
    failure (network blip, auth, quota) shouldn't take down a run that
    already succeeded locally -- the local file is the source of truth, Hub
    is just insurance against losing it if this box gets terminated."""
    if not ckpt_hub_repo:
        return
    local_path = Path(local_path)
    try:
        api = HfApi()
        api.create_repo(repo_id=ckpt_hub_repo, repo_type="dataset", token=hf_token, exist_ok=True, private=True)
        logging.info(f"Pushing checkpoint {local_path} -> {ckpt_hub_repo}:{local_path.name}...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=ckpt_hub_repo,
            repo_type="dataset",
            token=hf_token,
        )
        logging.info(f"Successfully pushed checkpoint to {ckpt_hub_repo}:{local_path.name}")
    except Exception as e:
        logging.warning(f"Could not push checkpoint {local_path} to Hub ({ckpt_hub_repo}:{local_path.name}): {e}")


def pull_ckpt_from_hub(local_path, ckpt_hub_repo, hf_token):
    """Best-effort download of a checkpoint (by basename) from ckpt_hub_repo
    into local_path, for resuming on a fresh box that lost local disk state.
    Returns True if local_path exists after this call, False otherwise (Hub
    repo/file doesn't exist yet, or the download failed for any reason --
    caller falls back to recomputing from scratch)."""
    if not ckpt_hub_repo:
        return False
    local_path = Path(local_path)
    try:
        logging.info(f"Local checkpoint {local_path} not found -- trying to pull "
                     f"{ckpt_hub_repo}:{local_path.name} from the Hub...")
        downloaded = hf_hub_download(
            repo_id=ckpt_hub_repo,
            filename=local_path.name,
            repo_type="dataset",
            token=hf_token,
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(downloaded, local_path)
        logging.info(f"Successfully pulled checkpoint from {ckpt_hub_repo}:{local_path.name} -> {local_path}")
        return True
    except Exception as e:
        logging.info(f"Could not pull checkpoint from Hub ({ckpt_hub_repo}:{local_path.name}): {e} -- will compute it instead.")
        return False


def load_mlb_classes(mlb_dir):
    """Load mlb classes for the labels. Uses these predefined classes in MultiLabel Binarization"""
    labels_file = os.path.join(mlb_dir, "labels_partition.json")
    logging.info(f"Loading MLB classes from {labels_file}")
    with open(labels_file, "r") as f:
        labels_partition = json.load(f)

    mlb_classes = labels_partition["classes"]

    return mlb_classes


def load_mlb_classes_and_clusters(mlb_dir):
    """Load mlb classes plus the flat label clustering built by
    `scripts/build_label_clusters.py` (`cluster_assignments.json`,
    `cluster_to_labels.json` in the same dir), for the full-scale
    candidate-generator subsystem. `cluster_ids` is index-aligned with
    `mlb_classes` (both derived from the same `labels_partition.json`)."""
    mlb_classes = load_mlb_classes(mlb_dir)

    cluster_assignments_file = os.path.join(mlb_dir, "cluster_assignments.json")
    logging.info(f"Loading label clusters from {cluster_assignments_file}")
    with open(cluster_assignments_file, "r") as f:
        cluster_assignments = json.load(f)
    cluster_ids = cluster_assignments["cluster_id"]
    num_clusters = cluster_assignments["num_clusters"]
    assert len(cluster_ids) == len(mlb_classes), (
        f"cluster_assignments.json has {len(cluster_ids)} entries, "
        f"expected {len(mlb_classes)} (one per label in labels_partition.json) "
        f"-- was build_label_clusters.py run against a stale labels_partition.json?"
    )

    cluster_to_labels_file = os.path.join(mlb_dir, "cluster_to_labels.json")
    with open(cluster_to_labels_file, "r") as f:
        cluster_to_labels = {int(k): v for k, v in json.load(f).items()}

    return mlb_classes, cluster_ids, num_clusters, cluster_to_labels


def sanity_check(dataset, tokenizer, mlb_classes, label_delim=None, num_samples=5):
    """
    Perform sanity checks on the processed dataset.
    
    Args:
        dataset: Processed Dataset object from onehotify
        tokenizer: Hugging Face tokenizer used in onehotify
        mlb_classes: List of predefined label classes
        label_delim: Delimiter used for splitting labels (default: None)
        num_samples: Number of samples to check (default: 5)
    
    Returns:
        None (prints results of sanity checks)
    """
    # Ensure we don't check more samples than available
    num_samples = min(num_samples, len(dataset))
    
    logging.info(f"Performing sanity check on {num_samples} samples...\n")
    
    for i in range(num_samples):
        sample = dataset[i]
        
        # Extract data
        input_ids = torch.tensor(sample['input_ids'])
        attention_mask = torch.tensor(sample['attention_mask'])
        one_hot_labels = torch.tensor(sample['one_hot_labels'])
        original_labels = sample['labels']  # Original string from the dataset
        
        # Check 1: Tokenization and decoding (truncate to 500 chars)
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)[:500]
        original_text = dataset[i]['text'][:500]
        print(f"Sample {i} - Decoded text (first 500 chars): {decoded_text}")
        print(f"Sample {i} - Original text (first 500 chars): {original_text}")
        
        # Check 2: One-hot labels vs original labels
        # Get indices where one_hot_labels is 1
        predicted_label_indices = torch.where(one_hot_labels == 1)[0]
        predicted_labels = [mlb_classes[idx] for idx in predicted_label_indices.tolist()]
        
        # Get original labels as a list
        if label_delim:
            original_label_list = original_labels.split(label_delim)
        else:
            original_label_list = [original_labels]
        
        logging.info(f"Sample {i} - Predicted labels from one-hot: {predicted_labels}")
        logging.info(f"Sample {i} - Original labels: {original_label_list}")
        
        # Verify that predicted labels match original labels
        labels_match = set(predicted_labels) == set(original_label_list)
        logging.info(f"Sample {i} - Labels match: {labels_match}")
        logging.info("-" * 50)


def transform_mutual_info(l2r_bootstrap, toks, lbs, output_dir, fname, ckpt_hub_repo=None, hf_token=None):
    """
    Retrieve and transform mutual information data into a PyTorch tensor.

    Args:
        l2r_bootstrap (dict): Dictionary containing mutual information data (key: 'mutual_info_jaccard').
        toks (list): List of tokens.
        lbs (list): List of labels.
        output_dir (str): Directory path for saving/loading transformed data.
        fname (str): Filename for saving/loading transformed data.
        ckpt_hub_repo (str, optional): HF Hub dataset repo to push/pull this checkpoint to/from.
        hf_token (str, optional): HF token for ckpt_hub_repo.

    Returns:
        torch.Tensor: Transformed mutual information scores as a PyTorch tensor on the default device.
    """
    # Retrieve mutual information data and validate shape
    info = l2r_bootstrap.get('mutual_info_jaccard', None)
    assert info.shape == (len(toks), len(lbs)), \
        f"Expected info shape {(len(toks), len(lbs))}, got {info.shape}"

    # Checkpoint the Box-Cox output: this and the two rank tables downstream
    # are what OOM-killed Stage 3 twice at Wiki10-31K full scale even after
    # the epoch loop finished cleanly. Without this, any failure past this
    # point forced redoing the whole ~7-8hr epoch loop just to get back here.
    bcx_ckpt_file = join_path_file(fname + '_bcx_mut_infos', Path(output_dir), ext='.pkl')
    if not bcx_ckpt_file.exists():
        pull_ckpt_from_hub(bcx_ckpt_file, ckpt_hub_repo, hf_token)
    if bcx_ckpt_file.exists():
        logging.info(f"Found {bcx_ckpt_file}. Loading Box-Cox-transformed mutual info (skipping Box-Cox transform)...")
        bcx_mut_infos = torch.load(bcx_ckpt_file)
    else:
        # Stays on CPU deliberately: downstream (compute_token_label_ranks /
        # compute_label_token_ranks -> argsort/stack/DataFrame -> to_feather) is
        # pure CPU-bound work. At Wiki10-31K full scale this tensor is ~16GB;
        # moving it to default_device() here used to OOM the GPU (already
        # ~89/94.5GB full from the preceding training/joint_pmf steps that don't
        # get freed) allocating the argsort+stack copies. System RAM has room
        # (333GB free), the GPU doesn't.
        bcx_mut_infos = torch.from_numpy(stat_transform(info, Path(output_dir), fname))
        logging.info(f"Saving Box-Cox-transformed mutual info checkpoint to {bcx_ckpt_file}...")
        torch.save(bcx_mut_infos, bcx_ckpt_file)
        logging.info(f"Successfully saved checkpoint to {bcx_ckpt_file}")
        generated_files.append((bcx_ckpt_file, "Box-Cox-transformed mutual information tensor (checkpoint)"))
        push_ckpt_to_hub(bcx_ckpt_file, ckpt_hub_repo, hf_token)

    assert bcx_mut_infos.shape == (len(toks), len(lbs)), \
        f"Expected bcx_mut_infos shape {(len(toks), len(lbs))}, got {bcx_mut_infos.shape}"

    return bcx_mut_infos


def compute_token_label_ranks(bcx_mut_infos, toks, lbs):
    """
    Compute mutual information scores and ranks for tokens per label, and store in a DataFrame.
    
    Args:
        bcx_mut_infos (torch.Tensor): Transformed mutual information scores as a PyTorch tensor on the default device.
        toks (list): List of tokens.
        lbs (list): List of labels.
        output_dir (str): Directory path for saving/loading transformed data.
        fname (str): Filename for saving/loading transformed data.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['token', 'label', 'bcx_mutual_info', 'rank'],
                      where each row represents a token-label pair with its mutual information
                      score and rank of the token for that specific label.
    """
    # Compute ranks of tokens for each label
    ranked = bcx_mut_infos.argsort(descending=True, dim=0).argsort(dim=0)

    # Build the long-format (token, label, bcx_mutual_info, rank) table
    # directly via numpy repeat/tile/reshape instead of the old wide
    # MultiIndex DataFrame -> stack(level=0) -> reset_index() pipeline. At
    # Wiki10-31K full scale (len(toks)=128256, len(lbs)=30938, ~3.97B rows)
    # that pipeline held several full-size copies simultaneously, and
    # torch.stack((bcx_mut_infos [float32], ranked [int64]), ...) silently
    # type-promoted everything to float64 -- together this was OOM-killed by
    # the Linux OOM-killer at 521/525GB system RAM. Row order here is
    # identical to before: stack(level=0)/reset_index() sorts by the
    # existing (token, label) integer index, which was already ascending, so
    # it's equivalent to a plain row-major flatten -- exactly what
    # repeat/tile below produce. No dtype promotion: rank stays int32 (ranks
    # top out at len(toks)-1 or len(lbs)-1, both far under int32 range) and
    # bcx_mutual_info stays its native (float32) precision instead of being
    # upcast to float64.
    n_tok, n_lbl = bcx_mut_infos.shape
    bcx_np = bcx_mut_infos.cpu().numpy()
    ranked_np = ranked.cpu().numpy().astype(np.int32)
    del ranked
    df_token_per_label_scores_ranks = pd.DataFrame({
        'token': np.repeat(np.arange(n_tok, dtype=np.int32), n_lbl),
        'label': np.tile(np.arange(n_lbl, dtype=np.int32), n_tok),
        'bcx_mutual_info': bcx_np.reshape(-1),
        'rank': ranked_np.reshape(-1),
    })

    # Validate DataFrame length
    assert len(df_token_per_label_scores_ranks) == len(toks) * len(lbs), \
        f"Expected len(df_token_per_label_scores_ranks) {len(toks) * len(lbs)}, got {len(df_token_per_label_scores_ranks)}"

    return df_token_per_label_scores_ranks


def compute_label_token_ranks(bcx_mut_infos, toks, lbs):
    """
    Compute mutual information scores and ranks for labels per token, and store in a DataFrame.
    
    Args:
        bcx_mut_infos (torch.Tensor): Transformed mutual information scores as a PyTorch tensor on the default device.
        toks (list): List of tokens.
        lbs (list): List of labels.
        output_dir (str): Directory path for saving/loading transformed data.
        fname (str): Filename for saving/loading transformed data.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['token', 'label', 'bcx_mutual_info', 'rank'],
                      where each row represents a token-label pair with its mutual information
                      score and rank of the label for that specific token.
    """
    # Compute ranks of labels for each token
    ranked = bcx_mut_infos.argsort(descending=True, dim=1).argsort(dim=1)

    # See compute_token_label_ranks above for why this builds the long-format
    # table directly via numpy instead of the old wide-DataFrame ->
    # stack(level=0) -> reset_index() pipeline (same OOM, same fix, dim=1
    # rank instead of dim=0).
    n_tok, n_lbl = bcx_mut_infos.shape
    bcx_np = bcx_mut_infos.cpu().numpy()
    ranked_np = ranked.cpu().numpy().astype(np.int32)
    del ranked
    df_label_per_token_scores_ranks = pd.DataFrame({
        'token': np.repeat(np.arange(n_tok, dtype=np.int32), n_lbl),
        'label': np.tile(np.arange(n_lbl, dtype=np.int32), n_tok),
        'bcx_mutual_info': bcx_np.reshape(-1),
        'rank': ranked_np.reshape(-1),
    })

    # Validate DataFrame length
    assert len(df_label_per_token_scores_ranks) == len(toks) * len(lbs), \
        f"Expected len(df_label_per_token_scores_ranks) {len(toks) * len(lbs)}, got {len(df_label_per_token_scores_ranks)}"
    
    return df_label_per_token_scores_ranks
    

# =============================================================================
# First Main Function - Data Processing and Information Calculation
# =============================================================================
def main_preprocessing(
    output_dir: str,
    fname: str,
    toks,
    lbs,
    label_delim: str = ";",
    workers: int = None,
    bs: int = 16,
    chnk_sz: int = 200,
    info=None,
    dsets=None,
    ckpt_hub_repo: str = None,
    hf_token: str = None,
):
    """
    Bootstrapping a learning-to-rank model - Part 1: Preprocessing and Information Calculation
    """
    global generated_files
    generated_files = []
    
    logging.info("\nOnehotifying the text and labels...")
    logging.info(f"Number of Tokens = {len(toks)}, Number of labels = {len(lbs)}")
    sanity_check(dsets, info.tokenizer, info.mlb_classes, label_delim=label_delim, num_samples=2)

    # -----------------------------------------------------------------------------
    # Data Chunking and PMF Computation
    # -----------------------------------------------------------------------------
    logging.info(f"chunking the labels into {info.lblsize/chnk_sz} many chunks, each chunk has {chnk_sz} many labels.")
    dls = info.create_chunked_dataloaders()
    info.sanity_check_chunked()

    logging.info(f"Computing joint pmf by batching {len(dsets)} datapoints into batches of size {bs} "
                 f"(so total {len(dsets)/bs} mini-batches). Also each mini-batch only has a chunk of the "
                 f"{chnk_sz} lbs (and we have {info.lblsize/chnk_sz} many dataloders).")
    
    # Load or compute joint probability mass function. This is the output of
    # the ~7-8hr epoch loop over all label chunks -- by far the most
    # expensive step in Stage 3, and until now the save call below was
    # commented out, so ANY failure downstream (Box-Cox, rank computation,
    # etc.) meant redoing the whole epoch loop from scratch on every retry.
    joint_pmf_file = join_path_file(fname + '_p_TL', Path(output_dir), ext='.pkl')
    if not joint_pmf_file.exists():
        pull_ckpt_from_hub(joint_pmf_file, ckpt_hub_repo, hf_token)
    if joint_pmf_file.exists():
        logging.info(f"Found {joint_pmf_file}. Loading it now...")
        p_TL = torch.load(joint_pmf_file)
    else:
        p_TL = info.joint_pmf()
        logging.info(f"Saving joint probability mass function to {joint_pmf_file}...")
        torch.save(p_TL, joint_pmf_file)
        logging.info(f"Successfully saved joint probability mass function to {joint_pmf_file}")
        generated_files.append((joint_pmf_file, "Joint probability mass function between tokens and labels"))
        push_ckpt_to_hub(joint_pmf_file, ckpt_hub_repo, hf_token)

    assert p_TL.shape == (info.toksize, info.lblsize, 2, 2), \
        f"Expected p_TL shape {(info.toksize, info.lblsize, 2, 2)}, got {p_TL.shape}"

    # Load or compute other information metrics (cheap to derive from p_TL,
    # but checkpointed too so a crash right after doesn't force even that
    # small amount of recomputation).
    info_pkl_file = join_path_file(fname + '_info', Path(output_dir), ext='.pkl')
    if not info_pkl_file.exists():
        pull_ckpt_from_hub(info_pkl_file, ckpt_hub_repo, hf_token)
    if info_pkl_file.exists():
        logging.info(f"Found {info_pkl_file}. Loading it...")
        p_T, p_L, p_TxL, H_T, H_L, I_TL = torch.load(info_pkl_file)
    else:
        p_T, p_L, p_TxL, H_T, H_L, I_TL = info.compute()
        logging.info(f"Saving information metrics to {info_pkl_file}...")
        torch.save((p_T, p_L, p_TxL, H_T, H_L, I_TL), info_pkl_file)
        logging.info(f"Successfully saved information metrics to {info_pkl_file}")
        generated_files.append((info_pkl_file, "Information-theoretic metrics (token/label probabilities, entropy, mutual information)"))
        push_ckpt_to_hub(info_pkl_file, ckpt_hub_repo, hf_token)
    
    # Validate tensor shapes
    assert p_TL.shape == (info.toksize, info.lblsize, 2, 2), \
        f"Expected p_TL shape {(info.toksize, info.lblsize, 2, 2)}, got {p_TL.shape}"
    assert p_T.shape == (info.toksize, 2, 1), \
        f"Expected p_T shape {(info.toksize, 2, 1)}, got {p_T.shape}"
    assert p_L.shape == (info.lblsize, 1, 2), \
        f"Expected p_L shape {(info.lblsize, 1, 2)}, got {p_L.shape}"
    assert p_TxL.shape == (info.toksize, info.lblsize, 2, 2), \
        f"Expected p_TxL shape {(info.toksize, info.lblsize, 2, 2)}, got {p_TxL.shape}"
    assert H_T.shape == torch.Size([info.toksize]), \
        f"Expected H_T shape {[info.toksize]}, got {H_T.shape}"
    assert H_L.shape == torch.Size([info.lblsize]), \
        f"Expected H_L shape {[info.lblsize]}, got {H_L.shape}"
    assert I_TL.shape == (info.toksize, info.lblsize), \
        f"Expected I_TL shape {(info.toksize, info.lblsize)}, got {I_TL.shape}"
    
    logging.info("Done computing joint pmf!")

    # -----------------------------------------------------------------------------
    # Data Validation
    # -----------------------------------------------------------------------------
    logging.info("Checking for nans/negs...")
    for k, v in dict(p_T=p_T, p_L=p_L, p_TxL=p_TxL, H_T=H_T, H_L=H_L, I_TL=I_TL).items():
        howmany = torch.where(v < 0, True, False).sum().item()
        negs = torch.where(v < 0, v, v.new_zeros(v.shape))
        logging.info(f'Checking nans/negs in {k}...')
        
        if howmany:
            logging.warning(f"Avg value of negs: {negs.sum()/howmany}")
            logging.warning("Those negs on an avg are pretty close to zero. So we need not worry. Let's roll!")
            test_fail(test_nanegs, kwargs={k: v}, contains=f'{k} has negs')
            torch.topk(v.flatten(), 10, largest=False)
        else:
            logging.info(f"{k} does not have any negs. Let's move on!")
        
        logging.info("*" * 50)

    # -----------------------------------------------------------------------------
    # Save Results
    # -----------------------------------------------------------------------------
    label_prob_file = join_path_file(fname + '_p_L', Path(output_dir), ext='.pkl')
    logging.info(f"Saving label probabilities to {label_prob_file}...")
    torch.save(p_L, label_prob_file)
    logging.info(f"Successfully saved label probabilities to {label_prob_file}")
    generated_files.append((label_prob_file, "Label probabilities tensor used for ranking calculation"))
    
    # Calculate metrics
    eps = I_TL.new_empty(1).fill_(1e-15)
    info_lbl_entropy = I_TL / (H_L + eps)
    info_jaccard = I_TL / (H_T.unsqueeze(-1) + H_L.unsqueeze(0) - I_TL + eps)
    
    assert not info_lbl_entropy.isnan().all()
    assert not info_jaccard.isnan().all()
    
    l2r_bootstrap = {
        'toks': toks,
        'lbs': lbs,
        'mut_info_lbl_entropy': info_lbl_entropy,
        'mutual_info_jaccard': info_jaccard
    }
    
    token_label_info_file = join_path_file(fname + '_tok_lbl_info', Path(output_dir), ext='.pkl')
    logging.info(f"Saving token-label information to {token_label_info_file}...")
    torch.save(l2r_bootstrap, token_label_info_file)
    logging.info(f"Successfully saved token-label information to {token_label_info_file}")
    generated_files.append((token_label_info_file, "Dictionary containing tokens, labels, and mutual information metrics"))
    push_ckpt_to_hub(token_label_info_file, ckpt_hub_repo, hf_token)

    # Print summary of generated files with descriptions
    print("\n" + "="*80)
    print("GENERATED FILES SUMMARY (PREPROCESSING)")
    print("="*80)
    for file_path, description in generated_files:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1]
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        # Format file size for human readability
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024*1024:
            size_str = f"{file_size/1024:.2f} KB"
        else:
            size_str = f"{file_size/(1024*1024):.2f} MB"
            
        print(f"File: {file_path}")
        print(f"  - Type: {file_ext[1:].upper()} file")
        print(f"  - Size: {size_str}")
        print(f"  - Description: {description}")
        print("-"*80)


# =============================================================================
# Second Main Function - Final Processing and Output
# =============================================================================
def main_processing_output(
    output_dir: str,
    fname: str,
    toks,
    lbs,
    ckpt_hub_repo: str = None,
    hf_token: str = None,
):
    """
    Bootstrapping a learning-to-rank model - Part 2: Final Processing and Output Generation
    """
    global generated_files
    generated_files = []

    # Create token and label DataFrames (trivial to (re)compute, but still
    # checked so a resumed run doesn't rewrite files that are already there)
    tokens_file = join_path_file(fname + '_tok', Path(output_dir), ext='.ft')
    if not tokens_file.exists():
        pull_ckpt_from_hub(tokens_file, ckpt_hub_repo, hf_token)
    if tokens_file.exists():
        logging.info(f"Found {tokens_file}. Skipping token-mapping save.")
    else:
        df_toks = pd.DataFrame([(i, w) for i, w in enumerate(toks)], columns=['token', 'tok_val'])
        logging.info(f"Saving token mappings to {tokens_file}...")
        df_toks.to_feather(tokens_file)
        logging.info(f"Successfully saved token mappings to {tokens_file}")
        generated_files.append((tokens_file, "DataFrame mapping token IDs to their string values"))
        push_ckpt_to_hub(tokens_file, ckpt_hub_repo, hf_token)

    labels_file = join_path_file(fname + '_lbl', Path(output_dir), ext='.ft')
    if not labels_file.exists():
        pull_ckpt_from_hub(labels_file, ckpt_hub_repo, hf_token)
    if labels_file.exists():
        logging.info(f"Found {labels_file}. Skipping label-mapping save.")
    else:
        df_lbs = pd.DataFrame([(i, w) for i, w in enumerate(lbs)], columns=['lbl', 'lbl_val'])
        logging.info(f"Saving label mappings to {labels_file}...")
        df_lbs.to_feather(labels_file)
        logging.info(f"Successfully saved label mappings to {labels_file}")
        generated_files.append((labels_file, "DataFrame mapping label IDs to their string values"))
        push_ckpt_to_hub(labels_file, ckpt_hub_repo, hf_token)

    # -----------------------------------------------------------------------------
    # Rank tables: the two expensive (~3.97B-row at Wiki10-31K full scale)
    # outputs that OOM-killed Stage 3 twice even after the epoch loop and
    # Box-Cox transform both completed cleanly. Each is checked/pulled before
    # computing and saved+pushed immediately (not batched at the end as
    # before), so (a) a crash on the second one doesn't lose the first, (b) a
    # resumed run that already has one or both doesn't recompute them, and
    # (c) peak memory only ever holds one ~63GB-scale DataFrame at a time
    # instead of both simultaneously. bcx_mut_infos (and the l2r_bootstrap
    # dict it's derived from) is only loaded/computed if at least one rank
    # table is actually still missing.
    token_per_label_rankings_file = join_path_file(fname + '_tok_rank_per_lbl', Path(output_dir), ext='.ft')
    label_per_token_rankings_file = join_path_file(fname + '_lbl_rank_per_tok', Path(output_dir), ext='.ft')
    if not token_per_label_rankings_file.exists():
        pull_ckpt_from_hub(token_per_label_rankings_file, ckpt_hub_repo, hf_token)
    if not label_per_token_rankings_file.exists():
        pull_ckpt_from_hub(label_per_token_rankings_file, ckpt_hub_repo, hf_token)

    if token_per_label_rankings_file.exists() and label_per_token_rankings_file.exists():
        logging.info(f"Found both {token_per_label_rankings_file} and {label_per_token_rankings_file}. "
                     "Skipping rank-table computation entirely (and the Box-Cox/mutual-info load that feeds it).")
    else:
        # Load and process the token-label information
        token_label_info_path = join_path_file(fname + '_tok_lbl_info', Path(output_dir), ext='.pkl')
        if not token_label_info_path.exists():
            pull_ckpt_from_hub(token_label_info_path, ckpt_hub_repo, hf_token)
        l2r_bootstrap = torch.load(
            token_label_info_path,
            map_location=torch.device('cpu')
        )

        # Retrieve and transform mutual information data
        bcx_mut_infos = transform_mutual_info(l2r_bootstrap, toks, lbs, output_dir, fname, ckpt_hub_repo, hf_token)

        if token_per_label_rankings_file.exists():
            logging.info(f"Found {token_per_label_rankings_file}. Skipping token-per-label rank computation.")
        else:
            df_token_per_label_scores_ranks = compute_token_label_ranks(bcx_mut_infos, toks, lbs)
            logging.info(f"Saving rankings of tokens per label to {token_per_label_rankings_file}...")
            df_token_per_label_scores_ranks.to_feather(token_per_label_rankings_file)
            logging.info(f"Successfully saved rankings of tokens per label to {token_per_label_rankings_file}")
            generated_files.append((token_per_label_rankings_file,
                                """
                                    DataFrame with columns ['token', 'label', 'bcx_mutual_info', 'rank'],
                                    where each row represents a token-label pair with its mutual information
                                    score and rank of the token for that specific label.
                                """))
            push_ckpt_to_hub(token_per_label_rankings_file, ckpt_hub_repo, hf_token)
            del df_token_per_label_scores_ranks

        if label_per_token_rankings_file.exists():
            logging.info(f"Found {label_per_token_rankings_file}. Skipping label-per-token rank computation.")
        else:
            df_label_per_token_scores_ranks = compute_label_token_ranks(bcx_mut_infos, toks, lbs)
            logging.info(f"Saving rankings of labels per token to {label_per_token_rankings_file}...")
            df_label_per_token_scores_ranks.to_feather(label_per_token_rankings_file)
            logging.info(f"Successfully saved rankings of labels per token to {label_per_token_rankings_file}")
            generated_files.append((label_per_token_rankings_file,
                                """
                                    DataFrame with columns ['token', 'label', 'bcx_mutual_info', 'rank'],
                                    where each row represents a token-label pair with its mutual information
                                    score and rank of the label for that specific token.
                                """))
            push_ckpt_to_hub(label_per_token_rankings_file, ckpt_hub_repo, hf_token)
            del df_label_per_token_scores_ranks

    # Print summary of generated files with descriptions
    print("\n" + "="*80)
    print("GENERATED FILES SUMMARY (PROCESSING OUTPUT)")
    print("="*80)
    for file_path, description in generated_files:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1]
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        # Format file size for human readability
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024*1024:
            size_str = f"{file_size/1024:.2f} KB"
        else:
            size_str = f"{file_size/(1024*1024):.2f} MB"
            
        print(f"File: {file_path}")
        print(f"  - Type: {file_ext[1:].upper()} file")
        print(f"  - Size: {size_str}")
        print(f"  - Description: {description}")
        print("-"*80)


# =============================================================================
# Unified Main Function that Calls Both Parts
# =============================================================================
def run_l2r_bootstrap_for_one_label_space(
    source, data, hub_model_id, base_dir, hf_token, output_dir, fname,
    label_delim, workers, bs, max_length, chnk_sz,
    mlb_classes_override=None,
    ckpt_hub_repo=None,
):
    """Runs the (previously main()-inlined) MIG bootstrap pipeline -- info
    construction, onehotify, main_preprocessing, main_processing_output --
    for one label vocabulary. `mlb_classes_override`, when given, restricts
    it to a single cluster's labels instead of the full vocabulary, so
    main()'s cluster loop can call this once per cluster with a distinct
    `fname` and get outputs that stay bounded by that cluster's label count."""
    # NOTE: workers=None here (not the actual `workers` param) matches the
    # pre-existing behavior this was refactored out of -- get_info() always
    # auto-detects via ifnone(None, min(8, num_cpus())) regardless of
    # --workers; only main_preprocessing() below honors the real value.
    info = get_info(source, data, hub_model_id, base_dir, hf_token, bs=bs, chnk_sz=chnk_sz, workers=None,
                     mlb_classes_override=mlb_classes_override)

    # Local-only checkpoint (not pushed to Hub -- it's a live Python object
    # dump of the tokenizer/dataset, a poor fit for cross-box restore, and
    # onehotify's tokenization is cheap next to the joint_pmf epoch loop).
    info_file = join_path_file(fname + '_info_obj', Path(output_dir), ext='.pkl')
    if info_file.exists():
        logging.info(f'Info object found in {output_dir}. Loading it...')
        info = torch.load(info_file)
        dsets = getattr(info, 'dsets', None)
    else:
        dsets = info.onehotify(max_length=max_length, label_delim=label_delim, overlap=10)
        logging.info(f"Saving info object to {info_file}...")
        torch.save(info, info_file)
        logging.info(f"Successfully saved info object to {info_file}")
        generated_files.append((info_file, "Information object containing tokenizer, dataset, and preprocessing details"))

    toks = info.tokenizer.convert_ids_to_tokens(range(len(info.tokenizer)))
    lbs = info.mlb_classes

    # Run preprocessing step
    main_preprocessing(
        output_dir=output_dir,
        fname=fname,
        toks=toks,
        lbs=lbs,
        label_delim=label_delim,
        workers=workers,
        bs=bs,
        chnk_sz=chnk_sz,
        info=info,
        dsets=dsets,
        ckpt_hub_repo=ckpt_hub_repo,
        hf_token=hf_token,
    )

    # Run final processing and output generation
    main_processing_output(output_dir, fname, toks, lbs, ckpt_hub_repo=ckpt_hub_repo, hf_token=hf_token)


def main(
    output_dir: str = "task2_output",
    source_url: str = "XURLs.MIMIC4_L2R",
    data: str = "mimic4_icd10_full",
    label_delim: str = ";",
    root_dir: str = "..",
    base_dir: str = None,
    workers: int = None,
    bs: int = 16,
    max_length: int = 512,
    chnk_sz: int = 200,
    logfile: str = "l2rboot",
    hub_model_id: str = "deb101/mistral-7b-mimic4",  # Task 1 Domain Adaptation model
    label_space: str = "label",
    ckpt_hub_repo: str = None,
):
    """
    Bootstrapping a learning-to-rank model (unified function that calls both parts).

    `label_space="cluster"` runs the whole bootstrap once per label cluster
    instead of once over the full vocabulary (Phase 4 of the full-scale
    candidate-generator plan) -- see run_l2r_bootstrap_for_one_label_space.
    """
    # -----------------------------------------------------------------------------
    # Data and File Setup
    # -----------------------------------------------------------------------------
    source = rank0_first(untar_xxx, eval(source_url))
    fname = '_'.join(data.split('_')[:-1])
    base_dir = base_dir + '/' + source_url.split('.')[1]
    output_dir = str(Path(base_dir) / output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger and print arguments
    logfile_path = setup_logging(**locals())
    print(f"You can tail the logfile: {logfile_path} to check logs")
    print_args(**locals())

    logging.info("****L2R Bootstraping for Multi-Label Classification Started****")

    # Hardcode your HF token (or load from environment variable)
    # hf_token = "your_hf_token_here"  # Replace or use os.environ["HF_TOKEN"]
    hf_token = os.environ["HF_TOKEN"]

    if label_space == "label":
        run_l2r_bootstrap_for_one_label_space(
            source, data, hub_model_id, base_dir, hf_token, output_dir, fname,
            label_delim, workers, bs, max_length, chnk_sz,
            ckpt_hub_repo=ckpt_hub_repo,
        )
    elif label_space == "cluster":
        mlb_classes_full, cluster_ids, num_clusters, cluster_to_labels = load_mlb_classes_and_clusters(base_dir)
        logging.info(f"Running the MIG bootstrap once per cluster ({num_clusters} clusters total, "
                     f"{len(mlb_classes_full)} labels overall).")
        for cluster_id in range(num_clusters):
            cluster_labels = [mlb_classes_full[i] for i in cluster_to_labels[cluster_id]]
            cluster_fname = f"{fname}_cluster{cluster_id}"
            logging.info(f"--- Cluster {cluster_id + 1}/{num_clusters}: {len(cluster_labels)} labels, "
                         f"fname prefix {cluster_fname!r} ---")
            run_l2r_bootstrap_for_one_label_space(
                source, data, hub_model_id, base_dir, hf_token, output_dir, cluster_fname,
                label_delim, workers, bs, max_length, chnk_sz,
                mlb_classes_override=cluster_labels,
                ckpt_hub_repo=ckpt_hub_repo,
            )
    else:
        raise ValueError(f"Unknown label_space {label_space!r}, expected 'label' or 'cluster'")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    # Convert the Namespace to a dictionary and pass to main via dictionary unpacking.
    main(**vars(args))