import lightning as L
import torch
from torch import Tensor
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Callable


def random_input_fn(num_samples, width, height):
    return torch.randint(0, 2, (num_samples, width, height))


class ToyDataset(Dataset):
    # target_fn outputs torch tensor
    def __init__(
        self,
        target_fn: Callable[[Tensor], Tensor],
        input_fn: Callable[[int, int, int], Tensor],
        num_samples=10000,
        width=8,
        height=8,
    ):
        self.width = width
        self.height = height

        self.inputs = input_fn(num_samples, width, height)

        self.targets = target_fn(self.inputs)

        self.inputs = self.inputs.unsqueeze(1).float()
        self.targets = self.targets.unsqueeze(1).float()

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


class ToyDataModule(L.LightningDataModule):
    def __init__(
        self,
        target_fn,
        input_fn=random_input_fn,
        batch_size: int = 32,
        num_samples: int = 10000,
        width: int = 8,
        height: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.target_fn = target_fn
        self.input_fn = input_fn
        self.num_samples = num_samples
        self.width = width
        self.height = height

    def setup(self, stage: str):
        self.full = ToyDataset(
            target_fn=self.target_fn,
            input_fn=self.input_fn,
            num_samples=self.num_samples,
            width=self.width,
            height=self.height,
        )

        self.train, self.val, self.test, self.predict = random_split(
            self.full,
            [
                int(self.num_samples * 0.7),
                int(self.num_samples * 0.1),
                int(self.num_samples * 0.1),
                int(self.num_samples * 0.1),
            ],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        pass
