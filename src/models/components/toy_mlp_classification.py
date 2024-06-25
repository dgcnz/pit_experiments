import torch


class ToyMLPClassification(torch.nn.Module):
    def __init__(
        self,
        img_width: int,
        img_height: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
    ):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.input_dim = img_width * img_height
        self.num_classes = num_classes
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_dim,
                out_features=hidden_dim,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=num_classes,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        predictions = self.layers(x)
        print(f"All x close: {torch.allclose(x[0], x[1])})")
        print(f"All predictions are equal?: {torch.all(predictions == predictions[0])}")
        return predictions
