import torch
from torch import nn
from torch import Tensor


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-12) -> None:
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # '-1' means last dimension. 
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta

        return out