from torch import Tensor
from timm.models import VisionTransformer


class PixelTransformer(VisionTransformer):
    def __init__(
        self,
        img_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        **kwargs
    ):
        super().__init__(
            img_size=img_size,
            patch_size=1,
            in_chans=1,
            num_classes=1,  # will be ignored
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            class_token=False,
            global_pool="",
            **kwargs,
        )

    def forward(self, x: Tensor):
        z = super().forward(x)
        z = z.view_as(x)
        return z
