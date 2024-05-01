import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch
from src.config import ConfigHolder
from src.model.custom_model import CustomModel


class CustomModelWrapper(L.LightningModule):
    def __init__(self, model: nn.Module, loss: nn.Module, optimizer: optim.Optimizer):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        # self.automatic_optimization = False
        # self.lr_schedulers = optim.lr_scheduler.StepLR(
        #     optimizer, step_size=1, gamma=0.1
        # )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, y, _, x = batch

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, y, _, x = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer
