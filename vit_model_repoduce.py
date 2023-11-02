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
        self.mlp_num_hiddens = mlp_num_hiddens
        self.mlp_num_outputs = mlp_num_outputs
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(mlp_num_hiddens, mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # linear --> gelu --> dropout --> linear --> dropout
        x = self.dropout1(self.gelu(self.dense1(x)))
        x = self.dropout2(self.dense2(x))
        return x        
###########


###########
# ViT block
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(norm_shape)
        self.MSA = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.layernorm2 = nn.LayerNorm(norm_shape)
        self.MLP = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, x):
        # from ViT paper, equation (2) - (3)
        x = self.MSA(*([self.layernorm1(x)]*3)) + x  # eq (2)
        x = self.MLP(self.layernorm2(x)) + x  # eq (3)
        return x  
###########


###########
# ViTMLP
# Patch embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        """
        num_hiddens is the feature size (d)
        """
        super().__init__()
        self.num_patches = (img_size//patch_size)**2
        self.conv = nn.Conv2d(3, num_hiddens,(patch_size, patch_size), stride=patch_size)

    def forward(self, x):
        x = self.conv(x)  # [B, d, img_size//patch_size, img_size//patch_size]
        # output shape should be: [B, num_patches, d]
        assert x.shape[-1]*x.shape[-2] == self.num_patches
        return x.flatten(2).permute(0,2,1)  
###########


###########
# ViT
class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.num_blks = num_blks
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        # create the cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        # because adding 1 cls token
        num_steps = self.patch_embedding.num_patches + 1   
        self.position_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        # the dropout after adding position embedding to patch embedding
        self.dropout = nn.Dropout(emb_dropout)  
        # create num_blks of vit block
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f'block {i}', 
                                 ViTBlock(num_hiddens, num_hiddens, mlp_num_hiddens, num_heads, blk_dropout, use_bias)
                                 )
        # the final MLP head is:
        # layrnorm --> linear
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, num_classes)
            )

    def forward(self, x):
        # patch embedding 
        x = self.patch_embedding(x)   # [B, num_patches, d]
        # concat cls token 
        cls = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, d]
        x = torch.cat((cls, x), dim = 1)
        # add position embedding, then dropout
        x = self.dropout(x + self.position_embedding)  # [B, num_patches+1, d]
        # num_blks of ViT block
        for blk in self.blks:
            x = blk(x)                                 # [B, num_patches+1, d]
        # only take cls token to the MLP head for classification
        x_cls = x[:, 0]   # [B, d]
        x = self.MLP_head(x_cls)
        return x
###########