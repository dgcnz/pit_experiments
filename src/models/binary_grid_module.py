from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy
import matplotlib.pyplot as plt
from src.utils.grokfast import gradfilter_ma, gradfilter_ema
from collections import namedtuple
from typing import Optional

GrokFastParams = namedtuple("GrokFastParams", ["alpha", "lamb"])


class BinaryGridLightningModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        lr_scheduler_interval: str = "epoch",
        grokfast_params: Optional[GrokFastParams] = None,
    ) -> None:
        """Initialize a `pl_classifier`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        # self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        # self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        # input image is (N, N)
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        if grokfast_params is not None:
            self.automatic_optimization = False
            self.gf_grads = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        if self.hparams.grokfast_params is not None:
            opt = self.optimizers()
            opt.zero_grad()
        loss, preds, targets = self.model_step(batch)
        self.last_train_step = {
            "batch": batch,
            "loss": loss,
            "preds": preds,
            "targets": targets,
        }
        if self.hparams.grokfast_params is not None:
            self.manual_backward(loss)
            self.gf_grads = gradfilter_ema(
                self.net,
                grads=self.gf_grads,
                alpha=self.hparams.grokfast_params.alpha,
                lamb=self.hparams.grokfast_params.lamb,
            )
            opt.step()
            # single scheduler
            sch = self.lr_schedulers()
            sch.step()

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        fig = self.plot_predictions(
            *self.last_train_step["batch"], preds=self.last_train_step["preds"]
        )
        self.logger.experiment.add_figure(
            "train/predictions", fig, global_step=self.current_epoch
        )

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.last_val_step = {
            "batch": batch,
            "loss": loss,
            "preds": preds,
            "targets": targets,
        }

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )
        fig = self.plot_predictions(
            *self.last_val_step["batch"], preds=self.last_val_step["preds"]
        )
        self.logger.experiment.add_figure(
            "val/predictions", fig, global_step=self.current_epoch
        )

    def plot_predictions(self, x: torch.Tensor, y: torch.Tensor, preds: torch.Tensor):
        fig, axes = plt.subplots(4, 3)
        for i in range(4):
            axes[i, 0].imshow(x[i, 0].cpu(), cmap="gray")
            axes[i, 1].imshow(preds[i][0].cpu().detach().numpy(), cmap="gray")
            axes[i, 2].imshow(y[i, 0].cpu(), cmap="gray")
            # title
            axes[i, 0].set_title("Input")
            axes[i, 1].set_title("Prediction")
            axes[i, 2].set_title("Target")
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
            axes[i, 2].axis("off")
        fig.tight_layout()
        return fig

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": self.hparams.lr_scheduler_interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
