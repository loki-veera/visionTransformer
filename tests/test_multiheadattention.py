"""Test multihead attention."""

import torch

from src.ViT.model.mhsa import MultiHeadSelfAttention


def test_attention():
    """Test the outputs of multihead attention."""
    torch.manual_seed(0)
    query = key = value = torch.randn(10, 50, 64)
    attention = MultiHeadSelfAttention(d_model=64, n_heads=8)
    output = attention(query, key, value)
    assert output.shape == query.shape
