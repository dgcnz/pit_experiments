import torch
import torch.nn as nn


class ToyViTDec(nn.Module):
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

        # Positional embeddings for decoder
        self.pos_embed = nn.Parameter(torch.zeros(1, self.img_size, self.embed_dim))

        # Patch to embedding conversion, used if input is image
        self.patch_to_embedding = nn.Linear(1, self.embed_dim)

        # Decoder part
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation="gelu",
            ),
            num_layers=num_blocks,
        )

        # Projection to output space
        self.fc_out = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        # Flatten the input and convert to embeddings
        x = x.view(batch_size, -1).unsqueeze(2)  # Assuming x is raw pixel data
        x = self.patch_to_embedding(x) + self.pos_embed

        # Generate a mask that prevents attention to future positions
        # attn_mask = torch.triu(
        #     torch.ones(self.img_size, self.img_size) * float("-inf"), diagonal=1
        # )

        # Decoder - autoregressively generating one pixel/patch at a time
        decoder_output = self.decoder(x, x)  # Self-attention

        # Output projection
        predictions = self.fc_out(decoder_output).squeeze(2)

        # Reshape predictions back to original shape
        predictions = predictions.view(
            batch_size, self.channels, self.img_height, self.img_width
        )

        return predictions
