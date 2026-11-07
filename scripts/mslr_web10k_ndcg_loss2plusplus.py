import torch
import torch.nn as nn
import torch.optim as optim
from ptranking.data.data_utils import get_data_loader
from ptranking.ltr_adhoc.eval.parameter import DataSetting
from torchmetrics.functional import retrieval_normalized_dcg

# Constants from ptranking
epsilon = 1e-8
LABEL_TYPE = type('LabelType', (), {'MultiLabel': 'MultiLabel', 'Permutation': 'Permutation'})()

# Helper functions from ptranking
def torch_dcg_at_k(batch_sorted_labels, cutoff=None, gpu=False, device=None):
    batch_gains = torch.pow(2.0, batch_sorted_labels) - 1.0
    batch_ranks = torch.arange(batch_gains.size(1), dtype=torch.float32, device=device)
    batch_discounts = 1.0 / torch.log2(batch_ranks + 2.0)
    batch_dcg = torch.sum(batch_gains * batch_discounts, dim=1)
    return batch_dcg

def ndcg_loss2plusplus_power_weights(batch_n_gains=None, discounts=None, mu=5.0, gpu=False, device=None):
    discounts = discounts.to(device) if device else discounts
    batch_n_gains = batch_n_gains.to(device) if device else batch_n_gains
    rho_ij = torch.abs(torch.pow(discounts[:, None], -1.0) - torch.pow(discounts[None, :], -1.0))
    torch_arange = torch.arange(batch_n_gains.size(1), dtype=torch.float32, device=device)
    ranks = torch_arange + 1.0
    abs_rank_deltas = torch.abs(ranks[:, None] - ranks[None, :]).type(torch.long).to(device)
    delta_ij = torch.abs(
        torch.pow(discounts[abs_rank_deltas - 1], -1.0) - torch.pow(discounts[abs_rank_deltas], -1.0)
    )
    delta_ij.diagonal().zero_()
    power_weights = (rho_ij + mu * delta_ij) * torch.abs(
        batch_n_gains[:, :, None] - batch_n_gains[:, None, :]
    )
    return power_weights

# Standalone NDCG_Loss2++ function
def ndcg_loss2plusplus(
    batch_preds, batch_stds, k=5, sigma=1.0, mu=5.0, label_type=LABEL_TYPE.MultiLabel, gpu=False, device=None, presort=False
):
    """
    Compute NDCG_Loss2++ loss given predicted scores and relevance labels.
    
    Args:
        batch_preds: Tensor of shape [batch_size, list_size] with predicted scores.
        batch_stds: Tensor of shape [batch_size, list_size] with relevance labels.
        k: Cutoff rank for truncation (default: 5).
        sigma: Sigmoid scaling factor (default: 1.0).
        mu: Weighting factor for NDCG_Loss2++ (default: 5.0).
        label_type: LABEL_TYPE.MultiLabel or Permutation (default: MultiLabel).
        gpu: Boolean indicating if CUDA is available (default: False).
        device: Torch device (cuda or cpu) (default: None).
        presort: If True, assume batch_stds is pre-sorted (default: False).
    
    Returns:
        batch_loss: Scalar tensor with the NDCG_Loss2++ loss.
    """
    assert label_type == LABEL_TYPE.MultiLabel

    # Move tensors to device
    batch_preds = batch_preds.to(device) if device else batch_preds
    batch_stds = batch_stds.to(device) if device else batch_stds

    # Sorting
    if presort:
        target_batch_preds, target_batch_stds = batch_preds, batch_stds
    else:
        target_batch_stds, batch_sorted_inds = torch.sort(batch_stds, dim=1, descending=True)
        target_batch_preds = torch.gather(batch_preds, dim=1, index=batch_sorted_inds)

    batch_preds_sorted, batch_preds_sorted_inds = torch.sort(target_batch_preds, dim=1, descending=True)
    batch_stds_sorted_via_preds = torch.gather(target_batch_stds, dim=1, index=batch_preds_sorted_inds)

    # Discounts
    batch_std_ranks = torch.arange(batch_preds_sorted.size(1), dtype=torch.float32, device=device)
    dists_1D = 1.0 / torch.log2(batch_std_ranks + 2.0)

    # Ideal DCG
    batch_idcgs = torch_dcg_at_k(batch_sorted_labels=target_batch_stds, gpu=gpu, device=device)

    # Gains and normalized gains
    batch_gains = torch.pow(2.0, batch_stds_sorted_via_preds) - 1.0
    batch_idcgs = torch.unsqueeze(batch_idcgs, dim=1)  # [batch_size] -> [batch_size, 1]
    batch_n_gains = batch_gains / batch_idcgs  # Broadcast [batch_size, list_size] / [batch_size, 1]

    # Power weights for NDCG_Loss2++
    power_weights = ndcg_loss2plusplus_power_weights(
        batch_n_gains=batch_n_gains, discounts=dists_1D, mu=mu, gpu=gpu, device=device
    )

    # Pairwise score differences
    batch_pred_diffs = (
        torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)
    ).clamp(min=-1e8, max=1e8)
    batch_pred_diffs[torch.isnan(batch_pred_diffs)] = 0.0

    # Weighted probabilities
    weighted_probas = (
        torch.sigmoid(sigma * batch_pred_diffs).clamp(min=epsilon) ** power_weights
    ).clamp(min=epsilon)
    log_weighted_probas = torch.log2(weighted_probas)

    # Truncation mask
    trunc_mask = torch.zeros(
        (batch_preds_sorted.shape[1], batch_preds_sorted.shape[1]), dtype=torch.bool, device=device
    )
    trunc_mask[:k, :k] = 1

    # Relevance difference mask
    batch_std_diffs = (
        torch.unsqueeze(batch_stds_sorted_via_preds, dim=2)
        - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)
    )
    padded_pairs_mask = batch_std_diffs > 0
    padded_log_weighted_probas = log_weighted_probas[padded_pairs_mask & trunc_mask]

    # Loss
    batch_loss = -torch.sum(padded_log_weighted_probas)
    return batch_loss

# Simple ranking model
class RankingModel(nn.Module):
    def __init__(self, input_dim):
        super(RankingModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single score per document
        )
    
    def forward(self, x):
        batch_size, list_size, input_dim = x.size()
        x = x.view(-1, input_dim)
        scores = self.model(x)
        return scores.view(batch_size, list_size)

# Initialize device and GPU flag
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = torch.cuda.is_available()
print(f"Using device: {device}, GPU: {gpu}")

# Load MSLR-WEB10K dataset
data_id = "MSLR10K"
data_setting = DataSetting(data_id=data_id)
train_loader = get_data_loader(data_id=data_id, split="train", batch_size=32)
test_loader = get_data_loader(data_id=data_id, split="test", batch_size=32)

# Initialize model and optimizer
feature_dim = 136  # MSLR-WEB10K has 136 features
model = RankingModel(input_dim=feature_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Smaller lr for stability

# Training parameters
epochs = 10
loss_params = {
    "k": 5,
    "sigma": 1.0,
    "mu": 5.0,
    "label_type": LABEL_TYPE.MultiLabel,
    "gpu": gpu,
    "device": device,
    "presort": False,
}

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in train_loader:
        features = batch["features"].to(device)
        relevance = batch["relevance"].to(device)

        # Validate relevance scores
        if torch.any(relevance < 0):
            raise ValueError("Relevance scores must be non-negative")

        # Normalize features
        features = torch.nn.functional.normalize(features, dim=-1)

        optimizer.zero_grad()
        scores = model(features)
        loss = ndcg_loss2plusplus(batch_preds=scores, batch_stds=relevance, **loss_params)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
total_ndcg = 0.0
num_batches = 0
with torch.no_grad():
    for batch in test_loader:
        features = batch["features"].to(device)
        relevance = batch["relevance"].to(device)

        # Validate relevance scores
        if torch.any(relevance < 0):
            raise ValueError("Relevance scores must be non-negative")

        # Normalize features
        features = torch.nn.functional.normalize(features, dim=-1)

        scores = model(features)
        ndcg = retrieval_normalized_dcg(preds=scores, target=relevance, top_k=5)
        total_ndcg += ndcg.mean().item()
        num_batches += 1

avg_ndcg = total_ndcg / num_batches
print(f"Average Test NDCG@5: {avg_ndcg:.4f}")