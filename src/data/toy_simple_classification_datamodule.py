import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class ToyDatasetSimpleClassification(Dataset):
    # batch_fn outputs torch tensor
    def __init__(
        self,
        batch_fn,
        num_samples=10000,
        width=8,
        height=8,
    ):
        # TODO: Currently num_samples is 1/2 of the actual resulting number of samples. Fix later
        self.width = width
        self.height = height

        self.inputs, self.targets = batch_fn(num_samples, width, height)

        self.inputs = self.inputs.unsqueeze(1).float()
        self.targets = self.targets.unsqueeze(1).float()

    def __getitem__(self, index):
        # print(index)
        # print(
        #     f"self.inputs_with_targets[index], self.target_labels[index]: {self.inputs_with_targets[index], self.target_labels[index]}"
        # )
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


class ToyDataModuleSimpleClassification(L.LightningDataModule):
    def __init__(
        self,
        batch_fn,
        batch_size: int = 32,
        num_samples: int = 10000,
        width: int = 8,
        height: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.batch_fn = batch_fn
        self.num_samples = num_samples
        self.width = width
        self.height = height

    def setup(self, stage: str):
        self.full = ToyDatasetSimpleClassification(
            batch_fn=self.batch_fn,
            num_samples=self.num_samples,
            width=self.width,
            height=self.height,
        )

        full_len = len(self.full)

        self.train, self.val, self.test, self.predict = random_split(
            self.full,
            [
                int(full_len * 0.7),
                int(full_len * 0.1),
                int(full_len * 0.1),
                int(full_len * 0.1),
            ],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        pass
