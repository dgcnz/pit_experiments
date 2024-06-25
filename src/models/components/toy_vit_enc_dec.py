import torch
import torch.nn as nn


class ToyViTEncDec(nn.Module):
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
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, self.img_size, self.embed_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_blocks,
        )

        # Decoder part
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, self.img_size, self.embed_dim))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_blocks,
        )

        # Projection to output space
        self.fc_out = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        # Flatten the input and project to embedding dimension
        x = x.view(batch_size, -1).unsqueeze(2)
        x_enc = self.patch_to_embedding(x)

        # Add positional embeddings
        x_enc = x_enc + self.pos_embed_enc

        # Encoder
        encoder_output = self.encoder(x_enc)

        # Decoder - uses encoded output as both tgt and memory (simple autoregressive model)
        x_dec = encoder_output + self.pos_embed_dec
        decoder_output = self.decoder(x_dec, encoder_output)

        # Output projection
        predictions = self.fc_out(decoder_output).squeeze(2)

        # Reshape predictions back to original shape
        predictions = predictions.view(
            batch_size, self.channels, self.img_height, self.img_width
        )

        return predictions
