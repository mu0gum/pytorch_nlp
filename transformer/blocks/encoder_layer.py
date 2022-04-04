import torch
from torch import nn
from torch import Tensor

from layers.multi_head_attention import MultiHeadAttention
from layers.layer_norm import LayerNorm
from layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, drop_prob: float, device: str) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, drop_prob=drop_prob, device=device)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        # self.dropout2 = nn.Dropout(p=drop_prob)

    # Code Reference : https://github.com/bentrevett/pytorch-seq2seq/
    def forward(self, x: Tensor, s_mask: Tensor = None) -> Tensor:
        # 1. compute self attention
        _x, _ = self.self_attention(query=x, key=x, value=x, mask=s_mask)

        # 2. dropout, residual connection and layer norm
        x = self.norm1(x + self.dropout(_x))

        # 3. feed forward
        _x = self.ffn(x)

        # 4
        x = self.norm2(x + self.dropout(_x))

        return x

    # Code Reference : https://github.com/nawnoes/pytorch-transformer
    # def forward(self, x, s_mask=None):

    #     norm_x = self.norm1(x)
    #     _x, _ = self.self_attention(query=norm_x, key=norm_x, value=norm_x, mask=s_mask)

    #     x = x + self.dropout(_x)

    #     norm_x = self.norm2(x)
    #     _x = self.ffn(norm_x)

    #     x = x + self.dropout(_x)

    #     return x


    # ahg : 로직이 조금씩 차이가 있는데, 성능 비교 해볼 것
    # def forward(self, x, s_mask=None):
    #     # for skip-connection
    #     _x = x
    #     # 1. compute self attention
    #     x = self.self_attention(q=x, k=x, v=x, mask=s_mask)
        
    #     # 2. add and layer normalization
    #     x = self.norm1(x + _x)
    #     x = self.dropout1(x)

    #     # for skip-connection
    #     _x = x
    #     # 3. positionwise feed forward network
    #     x = self.ffn(x)

    #     # 4. add and layer normalization
    #     x = self.norm2(x + _x)
    #     x = self.dropout2(x)

    #     return x
