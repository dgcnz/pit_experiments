import torch
import torch.nn as nn


class ToyViT(torch.nn.Module):
    def __init__(
        self,
        img_width: int,
        img_height: int,
        num_blocks=2,
        embed_dim=8,
        nhead=4,
        dim_feedforward=8,
    ):
        super().__init__()

        self.img_width = img_width
        self.img_height = img_height
        self.channels = 1
        self.img_size = img_width * img_height

        self.embed_dim = embed_dim

        self.patch_to_embedding = nn.Linear(1, self.embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.img_size, self.embed_dim)) # For CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, self.img_size, self.embed_dim))

        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                )
                for _ in range(num_blocks)
            ]
        )

        # self.fc = nn.Linear(in_features=self.embed_dim, out_features=1)
        self.fc = nn.Linear(
            in_features=self.img_size * self.embed_dim, out_features=self.img_size
        )

    def forward(self, x):
        input_shape = x.shape
        batch_size = input_shape[0]

        # Flatten the input and project to embedding dimension
        x = x.view(batch_size, -1).unsqueeze(2)
        x = self.patch_to_embedding(x)

        # Add CLS token and positional embeddings
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Get representations per token, excluding the CLS token
        # token_representations = x[:, 1:, :] # For CLS
        token_representations = x

        # predictions = self.fc(token_representations)
        predictions = self.fc(token_representations.reshape(batch_size, -1))

        # Reshape predictions back to original shape
        predictions = predictions.reshape(input_shape)

        return predictions
