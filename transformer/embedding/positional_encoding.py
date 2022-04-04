import math

import torch
from torch import nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, device: str) -> None:
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(seq_length, d_model, device=device)
        self.encoding.requires_grad = False  # positional encoding 값은 고정된 값 (학습 X)

        pos = torch.arange(0, seq_length, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        
        # # 논문 수식 그대로 (고현웅님 코드)
        # self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        # self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

        # 수치 계산 안전성을 위한 수식
        div_term = torch.exp(_2i * -(math.log(10000) / d_model))

        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)

        # inference test
        # self.register_buffer('positional_encoding', self.encoding)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length = x.size()

        return self.encoding[:seq_length, :]