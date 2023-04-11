import torch
import torch.nn as nn
from torch.nn.functional import softmax


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linear_q = nn.Linear(self.d_k, self.d_k)
        self.linear_k = nn.Linear(self.d_k, self.d_k)
        self.linear_v = nn.Linear(self.d_k, self.d_k)
        self.attention_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        bs = k.shape[0]
        tokens = k.shape[1]
        q = q.reshape(bs, tokens, self.n_heads, self.d_k)
        k = k.reshape(bs, tokens, self.n_heads, self.d_k)
        v = v.reshape(bs, tokens, self.n_heads, self.d_k)
        # Project the data using respective linear layers
        q = torch.einsum("ijkl -> ikjl", [self.linear_q(q)])
        k = torch.einsum("ijkl -> ikjl", [self.linear_k(k)])
        v = torch.einsum("ijkl -> ikjl", [self.linear_v(v)])
        # Compute scaled dot product attention
        attention_weights = self.scaled_dot_product_attention(q, k, v)

        # Concatenate and return the output
        concatenated_weights = attention_weights.reshape(bs, tokens, -1)
        out = self.attention_out(concatenated_weights)
        return out

    def scaled_dot_product_attention(self, q, k, v):
        # Compute the dot product
        dot_product = torch.einsum("...kl, ...rl -> ...kr", [q, k])
        # Scale the dot product
        dot_product = dot_product / (self.d_k) ** 0.5
        # Compute the softmax and multiply with values
        attention_weights = torch.einsum(
            "ijkl, ijlr -> ikjr", [softmax(dot_product, dim=-1), v]
        )
        return attention_weights
