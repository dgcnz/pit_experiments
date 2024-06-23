import torch


class ToyMLP(torch.nn.Module):
    def __init__(
        self,
        img_width: int,
        img_height: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.input_dim = img_width * img_height
        self.layers = torch.nn.Sequential(
            *[
                torch.nn.Linear(
                    in_features=self.input_dim if i == 0 else hidden_dim,
                    out_features=hidden_dim,
                )
                for i in range(num_layers)
            ],
            torch.nn.Linear(in_features=hidden_dim, out_features=self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        x = self.layers(x)
        x = x.view(-1, 1, self.img_height, self.img_width)
        return x
