import torch
import torch.nn as nn


class ToyTransformer(nn.Module):
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

        # Encoder part
        self.patch_to_embedding = nn.Linear(1, self.embed_dim)
        self.transformer = nn.Transformer(
            d_model=self.embed_dim,
            nhead=nhead,
            num_encoder_layers=num_blocks,
            num_decoder_layers=num_blocks,
            dim_feedforward=dim_feedforward,
        )

        # Projection to output space
        self.fc_out = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        # Flatten the input and project to embedding dimension
        x = x.view(batch_size, -1).unsqueeze(2)
        x_enc = self.patch_to_embedding(x)

        # Encoder
        transformer_output = self.transformer(
            src=x_enc,
            tgt=x_enc,
        )

        # Output projection
        predictions = self.fc_out(transformer_output).squeeze(2)

        # Reshape predictions back to original shape
        predictions = predictions.view(
            batch_size, self.channels, self.img_height, self.img_width
        )

        return predictions
