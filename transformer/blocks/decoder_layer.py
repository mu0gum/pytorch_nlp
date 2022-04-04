from torch import nn
from torch import Tensor

from layers.multi_head_attention import MultiHeadAttention
from layers.layer_norm import LayerNorm
from layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, drop_prob: float, device: str) ->  None:
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, drop_prob=drop_prob, device=device)
        self.norm1 = LayerNorm(d_model=d_model)
        # self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, drop_prob=drop_prob, device=device)
        self.norm2 = LayerNorm(d_model=d_model)
        # self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    # Code Reference : https://github.com/bentrevett/pytorch-seq2seq/
    def forward(self, dec: Tensor, enc: Tensor, t_mask: Tensor, s_mask: Tensor) -> tuple:
        # 1. compute self attention (masked)
        _x, _ = self.masked_self_attention(query=dec, key=dec, value=dec, mask=t_mask)

        # 2. dropout, residual connection and layer norm
        x = self.norm1(dec + self.dropout(_x))

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x, attention = self.enc_dec_attention(query=x, key=enc, value=enc, mask=s_mask)

            # 4. dropout, residual connection and layer norm
            x = self.norm2(x + self.dropout(_x))

        # 5. positionwise feed forward network
        _x = self.ffn(x)

        x = self.norm3(x + self.dropout(_x))

        return x, attention

    # # Code Reference : https://github.com/nawnoes/pytorch-transformer
    # def forward(self, dec, enc, t_mask, s_mask):
    #     norm_dec = self.norm1(dec)
    #     _x, _ = self.masked_self_attention(query=norm_dec, key=norm_dec, value=norm_dec, mask=t_mask)

    #     x = dec + self.dropout(_x)

    #     if enc is not None:
    #         norm_enc = self.norm2(enc)
    #         _x, attention = self.enc_dec_attention(query=x, key=norm_enc, value=norm_enc, mask=s_mask)

    #         x = x + self.dropout(_x)

    #     norm_x = self.norm3(x)
    #     x = self.ffn(x)

    #     x = x + self.dropout(x)

    #     return x, attention


    # ahg : 로직이 조금씩 차이가 있는데, 성능 비교 해볼 것
    # def forward(self, dec, enc, t_mask, s_mask):
    #     # 1. compute self attention (masked)
    #     _x = dec
    #     x = self.mask_mh_attention(q=dec, k=dec, v=dec, mask=t_mask)

    #     # 2. add and norm
    #     x = self.norm1(x + _x)
    #     x = self.dropout1(x)

    #     if enc is not None:
    #         # 3. compute encoder - decoder attention
    #         _x = x
    #         x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=s_mask)

    #         # 4. add and norm
    #         x = self.norm2(x + _x)
    #         x = self.dropout2(x)
        
    #     # 5. positionwise feed forward network
    #     _x = x
    #     x = self.ffn(x)

    #     # 6. add and norm
    #     x = self.norm3(x + _x)
    #     x = self.dropout3(x)

    #     return x

