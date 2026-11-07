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

# LambdaLoss class adapted from ptranking
class LambdaLoss(nn.Module):
    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(LambdaLoss, self).__init__()
        self.gpu = gpu
        self.device = device
        self.lambdaloss_dict = model_para_dict
        self.k, self.sigma, self.loss_type = (
            model_para_dict["k"],
            model_para_dict["sigma"],
            model_para_dict["loss_type"],
        )
        if "NDCG_Loss2++" == self.loss_type:
            self.mu = model_para_dict["mu"]
        # Scoring function
        self.scoring_function = nn.Sequential(
            nn.Linear(sf_para_dict["input_dim"], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        # Move all parameters to device
        self.to(device)

    def forward(self, batch_ranking):
        batch_size, list_size, input_dim = batch_ranking.size()
        x = batch_ranking.view(-1, input_dim)
        scores = self.scoring_function(x)
        return scores.view(batch_size, list_size)

    def inner_train(self, batch_preds, batch_stds, **kwargs):
        label_type = kwargs["label_type"]
        assert label_type == LABEL_TYPE.MultiLabel
        presort = kwargs.get("presort", False)

        if presort:
            target_batch_preds, target_batch_stds = batch_preds, batch_stds
        else:
            target_batch_stds, batch_sorted_inds = torch.sort(
                batch_stds, dim=1, descending=True
            )
            target_batch_preds = torch.gather(
                batch_preds, dim=1, index=batch_sorted_inds
            )

        batch_preds_sorted, batch_preds_sorted_inds = torch.sort(
            target_batch_preds, dim=1, descending=True
        )
        batch_stds_sorted_via_preds = torch.gather(
            target_batch_stds, dim=1, index=batch_preds_sorted_inds
        )

        batch_std_ranks = torch.arange(target_batch_preds.size(1), dtype=torch.float32, device=self.device)
        dists_1D = 1.0 / torch.log2(batch_std_ranks + 2.0)

        batch_idcgs = torch_dcg_at_k(
            batch_sorted_labels=target_batch_stds, gpu=self.gpu, device=self.device
        )

        batch_gains = torch.pow(2.0, batch_stds_sorted_via_preds) - 1.0
        # Reshape batch_idcgs to [batch_size, 1] for broadcasting
        batch_idcgs = torch.unsqueeze(batch_idcgs, dim=1)
        # Debug: Verify shapes
        print(f"batch_gains shape: {batch_gains.shape}, batch_idcgs shape: {batch_idcgs.shape}")
        batch_n_gains = batch_gains / batch_idcgs  # Now broadcastable

        if "NDCG_Loss2++" == self.loss_type:
            power_weights = ndcg_loss2plusplus_power_weights(
                batch_n_gains=batch_n_gains,
                discounts=dists_1D,
                mu=self.mu,
                gpu=self.gpu,
                device=self.device,
            )

        batch_pred_diffs = (
            torch.unsqueeze(batch_preds_sorted, dim=2)
            - torch.unsqueeze(batch_preds_sorted, dim=1)
        ).clamp(min=-1e8, max=1e8)
        batch_pred_diffs[torch.isnan(batch_pred_diffs)] = 0.0

        weighted_probas = (
            torch.sigmoid(self.sigma * batch_pred_diffs).clamp(min=epsilon)
            ** power_weights
        ).clamp(min=epsilon)
        log_weighted_probas = torch.log2(weighted_probas)

        trunc_mask = torch.zeros(
            (target_batch_preds.shape[1], target_batch_preds.shape[1]),
            dtype=torch.bool,
            device=self.device,
        )
        trunc_mask[: self.k, : self.k] = 1

        batch_std_diffs = (
            torch.unsqueeze(batch_stds_sorted_via_preds, dim=2)
            - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)
        )
        padded_pairs_mask = batch_std_diffs > 0
        padded_log_weighted_probas = log_weighted_probas[
            padded_pairs_mask & trunc_mask
        ]

        batch_loss = -torch.sum(padded_log_weighted_probas)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

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

# Model and loss parameters
sf_para_dict = {"input_dim": feature_dim}
model_para_dict = {
    "k": 5,
    "sigma": 1.0,
    "loss_type": "NDCG_Loss2++",
    "mu": 5.0,
}

# Initialize model
model = LambdaLoss(
    sf_para_dict=sf_para_dict, model_para_dict=model_para_dict, gpu=gpu, device=device
)

# Debug: Verify device placement
print(f"Features device: {features.device}")
print(f"Relevance device: {relevance.device}")
print(f"Model weights device: {model.scoring_function[0].weight.device}")

# Training step
model.train()
scores = model(features)  # Compute scores
batch_loss = model.inner_train(
    batch_preds=scores,
    batch_stds=relevance,
    label_type=LABEL_TYPE.MultiLabel,
    presort=False,
)

# Evaluate NDCG@5 using torchmetrics
ndcg_score = retrieval_normalized_dcg(preds=scores, target=relevance, top_k=5)

# Print results
print(f"Predicted Scores:\n{scores.detach().cpu().numpy()}")
print(f"Loss (NDCG_Loss2++): {batch_loss.item():.4f}")
print(f"NDCG@5: {ndcg_score.mean().item():.4f}")