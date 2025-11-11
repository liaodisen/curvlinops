
import numpy
import torch
from torch import nn

from curvlinops import EKFACLinearOperator, KFACLinearOperator, KFACInverseLinearOperator
from curvlinops.weighted_ce_loss import CrossEntropyLossWeighted

# make deterministic
torch.manual_seed(0)
numpy.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 10
D_in = 7
D_hidden = 5
C = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
idx = torch.arange(N).to(DEVICE)
y = torch.randint(low=0, high=C, size=(N,)).to(DEVICE)
y_with_idx = torch.stack((y, idx), dim=1)
data = [
    (torch.rand(N, D_in), y_with_idx)
]
wce = CrossEntropyLossWeighted(num_data=N)


model = nn.Sequential(
    nn.Linear(D_in, D_hidden),
    nn.ReLU(),
    nn.Linear(D_hidden, D_hidden),
    nn.Sigmoid(),
    nn.Linear(D_hidden, C),
).to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]

loss_function = wce.to(DEVICE)

EKFAC = EKFACLinearOperator(model, loss_function, params, data)
KFAC = KFACLinearOperator(model, loss_function, params, data)

inv_EKFAC = KFACInverseLinearOperator(EKFAC, use_exact_damping=True, damping=1e-3)
inv_KFAC = KFACInverseLinearOperator(KFAC, use_heuristic_damping=True, damping=1e-3)

D = KFAC.shape[0]
identity = torch.eye(D).to(DEVICE)

# create a random vector
v = torch.randn(D).to(DEVICE)

EKFAC_mat = inv_EKFAC @ v
KFAC_mat = inv_KFAC @ v

residual_norm = torch.linalg.norm(KFAC_mat - EKFAC_mat)
print(f"Residual norm: {residual_norm:.5f}")