"""Test the patching function."""

import torch

from src.ViT.train import create_patches


def test_patches():
    """Compute and test the patches."""
    x = torch.randn(20, 1, 28, 28)  # (batch_size, channels, height, width)
    output = create_patches(x)
    assert list(output.shape) == [20, 49, 16]  # (batch_size, n_patches, path_res)
