import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim



        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)



    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]



        V = self.values(values).view(N, value_len, self.num_heads, self.head_dim)

        K = self.keys(keys).view(N, key_len, self.num_heads, self.head_dim)
        
        Q = self.queries(query).view(N, query_len, self.num_heads, self.head_dim)



        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))



        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(N, query_len, self.embed_dim)
        return self.fc_out(out)




class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)



    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(self.dropout(attention) + query)
        forward = self.feed_forward(x)
        out = self.norm2(self.dropout(forward) + x)
        return out