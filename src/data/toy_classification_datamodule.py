import lightning as L
import torch
from torch import Tensor
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Callable


def random_input_fn(num_samples, width, height):
    return torch.randint(0, 2, (num_samples, width, height))


class ToyDatasetClassification(Dataset):
    # target_fn outputs torch tensor
    def __init__(
        self,
        target_fn: Callable[[Tensor], Tensor],
        input_fn: Callable[[int, int, int], Tensor],
        num_samples=10000,
        width=8,
        height=8,
    ):
        # TODO: Currently num_samples is 1/2 of the actual resulting number of samples. Fix later
        self.width = width
        self.height = height

        self.inputs = input_fn(num_samples, width, height)

        self.targets = target_fn(self.inputs)

        self.inputs = self.inputs.unsqueeze(1).float()
        self.targets = self.targets.unsqueeze(1).float()

        # Combine inputs with targets as images separated horizontally by a 0-valued line

        self.inputs_with_targets_positive = torch.cat(
            (self.inputs, torch.zeros(num_samples, 1, height, 1), self.targets), dim=3
        )

        self.inputs_with_targets_negative = torch.cat(
            (
                self.inputs,
                torch.zeros(num_samples, 1, height, 1),
                self.targets[torch.randint(0, num_samples, (num_samples,))],
            ),
            dim=3,
        )

        self.inputs_with_targets = torch.cat(
            (self.inputs_with_targets_positive, self.inputs_with_targets_negative),
            dim=0,
        )

        # Target labels: 0 for negative examples, 1 for positive examples
        self.target_labels = torch.cat(
            (torch.ones(num_samples), torch.zeros(num_samples)), dim=0
        )

        # TODO: Remove later. Set first pixel to 1 for positive examples, 0 for negative examples
        # self.inputs_with_targets[:, :, 0, 0] = self.target_labels[:, None]

        # print(f"self.inputs_with_targets.shape: {self.inputs_with_targets.shape}")
        # print(f"self.target_labels.shape: {self.target_labels.shape}")
        # print(f"self.target_labels: {self.target_labels}")

    def __getitem__(self, index):
        # print(index)
        # print(
        #     f"self.inputs_with_targets[index], self.target_labels[index]: {self.inputs_with_targets[index], self.target_labels[index]}"
        # )
        return self.inputs_with_targets[index], self.target_labels[index]

    def __len__(self):
        return len(self.target_labels)


class ToyDataModuleClassification(L.LightningDataModule):
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
        self.full = ToyDatasetClassification(
            target_fn=self.target_fn,
            input_fn=self.input_fn,
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
