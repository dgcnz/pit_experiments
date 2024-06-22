import lightning as L
import torch
from torch.utils.data import random_split, DataLoader, Dataset


# TODO: Make sure that all samples are unique.
class ToyDataset(Dataset):
    def __init__(self, target_fn, num_samples=10000, width=8, height=8):
        self.width = width
        self.height = height

        self.inputs = torch.randint(0, 2, (num_samples, self.width, self.height))
        self.targets = target_fn(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


class ToyDataModule(L.LightningDataModule):
    def __init__(
        self, target_fn, batch_size: int = 32, num_samples=10000, width=8, height=8
    ):
        super().__init__()
        self.batch_size = batch_size
        self.target_fn = target_fn
        self.num_samples = num_samples
        self.width = width
        self.height = height

    def setup(self, stage: str):
        self.full = ToyDataset(
            target_fn=self.target_fn,
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
