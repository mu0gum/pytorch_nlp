# from math import math
import math

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


# class TokenEmbedding(nn.Embedding):
#     def __init__(self, vocab_size, d_model):
#         super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

#         # super(TokenEmbedding, self).__init__()
#         # # pytorch torch.nn.Embedding() 사용
#         # # self.embedding = nn.Embedding(vocab_size, d_model, padding_size=1)
#         # self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)
#         # # self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
#         # self.d_model = d_model

#     # def forward(self, x):
#     #     # return self.embedding(x)
#     #     return self.embedding(x) * math.sqrt(self.d_model)