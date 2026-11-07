import torch
import torch.nn as nn
import torch.optim as optim
from ptranking.ltr_adhoc.listwise.approxNDCG import approxNDCG_loss
from ptranking.ltr_adhoc.eval.eval_utils import LABEL_TYPE
from torchmetrics.functional import retrieval_normalized_dcg

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
relevance = torch.tensor([[2.0, 0.0, 1.0, 0.0, 3.0], [1.0, 0.0, 0.0, 2.0, 1.0]], dtype=torch.float32)  # (2, 5)

# Initialize device and GPU flag
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = torch.cuda.is_available()  # Set gpu=True if CUDA is available
print(f"Using device: {device}, GPU: {gpu}")

# Move model and tensors to device
model = RankingModel(input_dim=feature_dim).to(device)
features = features.to(device)
relevance = relevance.to(device)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training step
model.train()
optimizer.zero_grad()
scores = model(features)  # Predict scores
loss = approxNDCG_loss(
    batch_preds=scores,
    batch_stds=relevance,
    alpha=10.0,
    label_type=LABEL_TYPE.MultiLabel,
    gpu=gpu
)  # Compute approxNDCG_loss
loss.backward()  # Backpropagate
optimizer.step()

# Evaluate NDCG@5 using torchmetrics
ndcg_score = retrieval_normalized_dcg(preds=scores, target=relevance, top_k=5)

# Print results
print(f"Predicted Scores:\n{scores.detach().cpu().numpy()}")
print(f"Loss (approxNDCG_loss): {loss.item():.4f}")
print(f"NDCG@5: {ndcg_score.mean().item():.4f}")