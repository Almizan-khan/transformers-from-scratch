import torch
import torch.nn as nn

from Encoder import MultiHeadAttention, TransformerBlock
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out