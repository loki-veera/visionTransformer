import torch
import torch.nn as nn

from .mhsa import MultiHeadSelfAttention


class ViT(nn.Module):
    def __init__(self, d_model, n_patches, n_encoders, n_heads, patch_res) -> None:
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

    def forward(self, x):
        linear_maps = self.linear_embed(x)
        inputs = torch.cat(
            (self.class_token.expand(x.shape[0], 1, -1), linear_maps), dim=1
        )
        inputs += self.pos_embed.repeat(x.shape[0], 1, 1)
        output = inputs
        for encoder in self.encoders:
            output = encoder(output)
        output = self.final_linear(output[:, 0])
        return nn.functional.softmax(output, dim=-1)

    def __positional_encoding(self, max_seq_length, d_model):
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
    def __init__(self, d_model, n_heads) -> None:
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

    def forward(self, x):
        x = self.layer_norm(x + self.attention(x, x, x))
        x = self.layer_norm(x + self.linear_layer(x))
        return x
