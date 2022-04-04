from torch import nn
from torch import Tensor

from blocks.encoder_layer import EncoderLayer


class Encoder(nn.Module):

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, n_layers: int, drop_prob: float, device: str) -> None:
        super(Encoder, self).__init__()
      
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob,
                                                  device=device) 
                                    for _ in range(n_layers)])

    def forward(self, x: Tensor, s_mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, s_mask)

        return x