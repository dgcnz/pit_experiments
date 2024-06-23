import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer


class LinearWarmupScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        """
        Creates a scheduler for a linear warmup.

        :param optimizer: Optimizer linked to the scheduler.
        :param warmup_steps: Number of steps over which to linearly warmup the learning rate.
        :param last_epoch: The index of the last epoch. This parameter is used when resuming a training job.
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        """
        Defines the learning rate schedule.

        :param step: Current step.
        :return: Multiplicative factor for the learning rate.
        """
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return 1.0
