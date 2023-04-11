import torch
import torch.nn as nn

from .mhsa import MultiHeadSelfAttention


class ViT(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_patches: int,
        n_encoders: int,
        n_heads: int,
        patch_res: int,
    ) -> None:
        """Initialize the transformer model.

        Args:
            d_model (int): Model dimension.
            n_patches (int): Number of patches of an image.
            n_encoders (int): Number of encoders.
            n_heads (int): Number of attention heads.
            patch_res (int): Resolution of each patch.
        """
        super().__init__()
        self.linear_embed = nn.Linear(patch_res * patch_res, d_model)
        self.class_token = nn.Parameter(torch.rand(1, d_model))
        self.register_buffer(
            "pos_embed", self.__positional_encoding(n_patches**2 + 1, d_model)
        )
        self.encoders = nn.ModuleList(
            [ViTEncoder(d_model, n_heads) for _ in range(n_encoders)]
        )
        self.final_linear = nn.Linear(d_model, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the model.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            torch.tensor: Softmax probabilities of classes.
        """
        linear_maps = self.linear_embed(x)
        inputs = torch.cat(
            (self.class_token.expand(x.shape[0], 1, -1), linear_maps), dim=1
        )
        pos_embeds = self.pos_embed.repeat(x.shape[0], 1, 1)
        inputs += pos_embeds
        output = inputs
        for encoder in self.encoders:
            output = encoder(output)
        output = self.final_linear(output[:, 0])
        return nn.functional.softmax(output, dim=-1)

    def __positional_encoding(self, max_seq_length: int, d_model: int) -> torch.Tensor:
        """Get the positional encoding values.

        Args:
            max_seq_length (int): Maximum sequence length.
            d_model (int): Model dimension.

        Returns:
            torch.tensor: Positional encoding values.
        """
        pos_matrix = torch.zeros(max_seq_length, d_model, requires_grad=False)
        for seq_index in range(0, max_seq_length):
            for index in range(0, d_model // 2):
                denom = torch.Tensor([seq_index / (10000 ** ((2 * index) / d_model))])
                (
                    pos_matrix[seq_index, 2 * index],
                    pos_matrix[seq_index, 2 * index + 1],
                ) = torch.sin(denom), torch.cos(denom)
        return pos_matrix.unsqueeze(0)


class ViTEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        """Initialize the encoder block.

        Args:
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
        """
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder.

        Args:
            x (torch.tensor): Input embedding for the encoder.

        Returns:
            torch.tensor: Output of the encoder.
        """
        x = self.layer_norm(x + self.attention(x, x, x))
        x = self.layer_norm(x + self.linear_layer(x))
        return x
