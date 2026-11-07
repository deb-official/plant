import torch
import torch.nn as nn
import torch.optim as optim
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
    batch_n_gains = batch_gains / batch_idcgs  # Broadcast [2, 5] / [2, 1]

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

# Synthetic data
batch_size = 2
list_size = 5
feature_dim = 10

features = torch.randn(batch_size, list_size, feature_dim)  # (2, 5, 10)
relevance = torch.tensor(
    [[2.0, 0.0, 1.0, 0.0, 3.0], [1.0, 0.0, 0.0, 2.0, 1.0]], dtype=torch.float32
)  # (2, 5)

# Initialize device and GPU flag
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = torch.cuda.is_available()
print(f"Using device: {device}, GPU: {gpu}")

# Validate relevance scores
if torch.any(relevance < 0):
    raise ValueError("Relevance scores must be non-negative")

# Normalize features and ensure device placement
features = torch.nn.functional.normalize(features, dim=-1).to(device)
relevance = relevance.to(device)

# Initialize model and optimizer
model = RankingModel(input_dim=feature_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training step
model.train()
optimizer.zero_grad()
scores = model(features)  # Compute scores
batch_loss = ndcg_loss2plusplus(
    batch_preds=scores,
    batch_stds=relevance,
    k=5,
    sigma=1.0,
    mu=5.0,
    label_type=LABEL_TYPE.MultiLabel,
    gpu=gpu,
    device=device,
    presort=False
)
batch_loss.backward()
optimizer.step()

# Evaluate NDCG@5 using torchmetrics
ndcg_score = retrieval_normalized_dcg(preds=scores, target=relevance, top_k=5)

# Print results
print(f"Predicted Scores:\n{scores.detach().cpu().numpy()}")
print(f"Loss (NDCG_Loss2++): {batch_loss.item():.4f}")
print(f"NDCG@5: {ndcg_score.mean().item():.4f}")