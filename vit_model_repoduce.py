import torch
from torch import nn
import math


###########
# scaled dot-product


class ScaledDotProductAttention(nn.Module):  
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, n, d)
    # Shape of keys: (batch_size, m, d)
    # Shape of values: (batch_size, m, v)
    
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        assert keys.shape[-2] == values.shape[-2], "the key and value did not have the same token length"
        # calcuate queries dot product with keys'transpose
        queries_dot_keys = queries@keys.permute(0,2,1)    # [B, n, m]
        # scale and softmax it
        softmax_qk = nn.functional.softmax(queries_dot_keys/math.sqrt(d), dim = -1) # [B, n, m]
        output = softmax_qk@values   # [B, n, v]
        return output
    
###########


###########
# multi-head attention 
class MultiHeadAttention(nn.Module): 
    """Multi-head attention.
    num_hiddens (d) is the d_model, the feature size
    num_heads (h) is the number of heads, 
    """
    # Shape of queries: (batch_size, n, d)
    # Shape of keys: (batch_size, m, d)
    # Shape of values: (batch_size, m, d)
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.attention = ScaledDotProductAttention(dropout)

        # nn.Linear(num_hiddens, num_hiddens, bias = bias) will also work
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)    # [d,d]
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)    # [d,d]
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)    # [d,d]
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)    # [d,d]

    def forward(self, queries, keys, values):
        # after linear projection: [batch_size, n or m, d]
        q, k ,v = self.W_q(queries), self.W_k(keys), self.W_v(values)
        assert k.shape[-2] == v.shape[-2], 'check the shapes of keys and values'
        # Transposing each q,k,v to the shape [batch_size * h, n or m, d / h]
        q,k,v = self.transposing_qkv(q), self.transposing_qkv(k), self.transposing_qkv(v)
        output = self.attention(q,k,v) # [B*h, n, d/h]
        output = self.transposing_output(output)
        return self.W_o(output)

        
    def transposing_qkv(self, x):
        B, n, d = x.shape
        # first reshape x into shape [B, n, h, d/h], then permute to [B, h, n ,d/h]
        h = self.num_heads
        assert d % h == 0, 'check number of heads and number of hiddens'
        x = x.reshape(B, n, h, d//h).permute(0,2,1,3) # [B, h, n ,d/h]
        return x.reshape(B*h, n ,d//h)
    
    def transposing_output(self, x):
        h = self.num_heads
        d = self.num_hiddens
        Bh, n, dk = x.shape # [B*h, n, d/h]
        # first make [B, h, n, d/h], then [B, n, h, d/h], finally [B, n, d]
        x = x.reshape(-1, h, n, dk).permute(0,2,1,3)
        return x.reshape(-1, n, d)    
###########


###########
# ViTMLP
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        pass

    def forward(self, x):
        pass
###########


###########
# encoder block

###########


###########
# ViT

###########