import math

from torch import nn
from torch import Tensor

from embedding.positional_encoding import PositionalEncoding
from embedding.token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, seq_length:int, drop_prob: float, device: str)-> None:
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, seq_length, device)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(x)

        return self.dropout(token_embedding + position_embedding)