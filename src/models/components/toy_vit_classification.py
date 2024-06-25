import einops
import torch
import torch.nn as nn


# class ToyViTClassification(torch.nn.Module):
#     def __init__(
#         self,
#         img_width: int,
#         img_height: int,
#         num_blocks: int,
#         embed_dim: int,
#         nhead: int,
#         dim_feedforward: int,
#         num_classes: int,
#         is_simple_classification: bool = False,
#     ):
#         super().__init__()

#         self.img_width = img_width
#         self.img_height = img_height
#         self.channels = 1
#         self.img_size = img_width * img_height
#         self.input_size = (
#             self.img_size * 2 + self.img_height
#             if not is_simple_classification
#             else self.img_size
#         )

#         self.embed_dim = embed_dim

#         self.patch_to_embedding = nn.Linear(1, self.embed_dim)

#         self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
#         self.pos_embed = nn.Parameter(
#             torch.randn(1, 1 + self.input_size, self.embed_dim)
#         )  # For CLS
#         # self.pos_embed = nn.Parameter(torch.zeros(1, self.img_size, self.embed_dim))

#         self.dropout = nn.Dropout(p=0.1)

#         # Based on https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
#         # encoder_layer = nn.TransformerEncoderLayer(
#         #     d_model=self.embed_dim,
#         #     nhead=nhead,
#         #     dim_feedforward=dim_feedforward,
#         # )
#         # self.transformer_encoder = nn.TransformerEncoder(
#         #     encoder_layer, num_layers=num_blocks
#         # )

#         self.transformer_encoder = nn.Sequential(
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.ReLU(),
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.ReLU(),
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.ReLU(),
#         )

#         self.fc = nn.Linear(in_features=self.embed_dim, out_features=num_classes)

#     def forward(self, x):
#         print(f"All x close from data?: {torch.allclose(x[0], x[1])})")
#         input_shape = x.shape
#         batch_size = input_shape[0]

#         # Flatten the input and project to embedding dimension
#         x = x.view(batch_size, -1).unsqueeze(2)
#         x = self.patch_to_embedding(x)
#         print(f"All x after embedding?: {torch.allclose(x[0], x[1])})")

#         # Add CLS token and positional embeddings
#         # cls_tokens = einops.repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
#         # Used some code from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/pit.py
#         cls_tokens = einops.repeat(self.cls_token, "() n d -> b n d", b=batch_size)

#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embed[:, : x.shape[1] + 1]
#         x = self.dropout(x)

#         # Transformer Blocks
#         x = self.transformer_encoder(x)

#         # predictions = self.fc(x[:, 0])
#         predictions = x[:, 1, 0].unsqueeze(1)
#         print(f"All x close after transformer: {torch.allclose(x[0], x[1])})")
#         print(
#             f"All predictions are equal?: {torch.allclose(predictions[0], predictions[1])}"
#         )

#         return predictions

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ToyViTClassification(torch.nn.Module):
    def __init__(
        self,
        img_width: int,
        img_height: int,
        num_blocks: int,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_classes: int,
        is_simple_classification: bool = False,
    ):
        super().__init__()

        self.img_width = img_width
        self.img_height = img_height
        self.channels = 1
        self.img_size = img_width * img_height
        self.input_size = (
            self.img_size * 2 + self.img_height
            if not is_simple_classification
            else self.img_size
        )

        self.embed_dim = embed_dim
        patch_height = 1
        patch_width = 1
        dim = embed_dim
        channels = 1

        assert (
            img_height % patch_height == 0 and img_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=img_height // patch_height,
            w=img_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(
            dim,
            depth=num_blocks,
            heads=nhead,
            dim_head=dim // nhead,
            mlp_dim=dim_feedforward,
        )

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)
