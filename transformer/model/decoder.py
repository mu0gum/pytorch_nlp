from torch import nn
from torch import Tensor

from blocks.decoder_layer import DecoderLayer


class Decoder(nn.Module):

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, n_layers: int, drop_prob: float, device: str) -> None:
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob,
                                                  device=device) 
                                    for _ in range(n_layers)])
    
    def forward(self, trg: Tensor, enc_src: Tensor, trg_mask: Tensor, src_mask: Tensor) -> tuple:
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        return trg, attention