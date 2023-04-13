"""Test model forward pass."""

import torch

from src.ViT.model.vit import ViT, ViTEncoder


def test_model():
    """Test the model dummy forward pass."""
    x = torch.randn((20, 49, 16))  # (bacth_size, n_patches, patch_size)
    model = ViT(
        d_model=64, n_patches=7, n_encoders=2, n_heads=8, patch_res=4, n_channels=1
    )
    preds = model(x)
    assert list(preds.shape) == [20, 10]


def test_encoder():
    """Test the model encoder."""
    x = torch.randn((20, 50, 64))  # (bacth_size, n_patches+1, embedding_size)
    encoder = ViTEncoder(d_model=64, n_heads=8)
    output = encoder(x)
    assert list(output.shape) == [20, 50, 64]
