import torch
from torch import nn
from torch import Tensor

from embedding.transformer_embedding import TransformerEmbedding
from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx: int, trg_pad_idx: int, enc_voc_size: int, dec_voc_size: int, d_model: int, n_head: int, max_len: int,
                ffn_hidden: int, n_layers: int, drop_prob: float, device: str) -> None:
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.enc_embedding = TransformerEmbedding(d_model=d_model,
                                                seq_length=max_len,
                                                vocab_size=enc_voc_size,
                                                drop_prob=drop_prob,
                                                device=device)

        self.dec_embedding = TransformerEmbedding(d_model=d_model,
                                                seq_length=max_len,
                                                vocab_size=dec_voc_size,
                                                drop_prob=drop_prob,
                                                device=device)

        self.encoder = Encoder(d_model=d_model,
                               ffn_hidden=ffn_hidden,
                               n_head=n_head,
                               n_layers=n_layers,
                               drop_prob=drop_prob,
                               device=device
                               )
        
        self.decoder = Decoder(d_model=d_model,
                               ffn_hidden=ffn_hidden,
                               n_head=n_head,
                               n_layers=n_layers,
                               drop_prob=drop_prob,
                               device=device
                               )
        
        self.linear = nn.Linear(d_model, dec_voc_size)
        
        # two linear layers test
        # self.linear1 = nn.Linear(d_model, d_model*4)
        # self.linear2 = nn.Linear(d_model*4, dec_voc_size)

    def make_src_mask(self, src: Tensor) -> Tensor:
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg: Tensor) -> Tensor:
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src: Tensor, trg: Tensor) -> tuple:
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]

        src = self.enc_embedding(src)
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]

        trg = self.dec_embedding(trg)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        output = self.linear(output)
        
        # two linear layers test
        # output = self.linear1(output)
        # output = self.linear2(output)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention