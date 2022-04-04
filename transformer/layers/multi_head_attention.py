import torch
from torch import nn
from torch import Tensor

from layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, drop_prob: float, device: str) -> None:
        super().__init__()
        
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        
        self.fc_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> tuple:
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.d_model)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention


# class MultiHeadAttention(nn.Module):

#     def __init__(self, d_model, n_head):
#         super(MultiHeadAttention, self).__init__()
#         self.n_head = n_head
#         self.attention = ScaleDotProductAttention()
#         self.w_q = nn.Linear(d_model, d_model)
#         self.w_k = nn.Linear(d_model, d_model)
#         self.w_v = nn.Linear(d_model, d_model)
#         self.w_concat = nn.Linear(d_model, d_model)
        
#         # inference test
#         # self.dropout = nn.Dropout(p=0.1)

#     def forward(self, q, k, v, mask=None):
#         # 1. dot product with weight matrices
#         q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

#         # 2. split tensor by number of heads
#         q, k, v = self.split(q), self.split(k), self.split(v)

#         # 3. do scale dot product to compute similarity
#         out, attention = self.attention(q, k, v, mask=mask)

#         # 4. concat and pass to linear layer
#         out = self.concat(out)
#         out = self.w_concat(out)

#         return out

#     def split(self, tensor):
#         batch_size, length, d_model = tensor.size()

#         assert d_model % self.n_head == 0

#         d_tensor = d_model // self.n_head
#         # 원소의 수를 유지하면서 텐서의 크기 변경(view)
#         # 1. batch_size, length, n_head, d_tensor 순서로 reshape (n_head * d_tensor = d_model)
#         # 2. length <-> n_head transpose (attention 연산할 때 모양을 맞춰주기 위해)
#         tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

#         return tensor
    
#     def concat(self, tensor):
#         batch_size, head, length, d_tensor = tensor.size()
#         d_model = head * d_tensor

#         # 1. n_head <-> length
#         # 2. batch_size. length, d_model 순서로 reshape
#         tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        
#         return tensor