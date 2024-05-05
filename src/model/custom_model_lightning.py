import os
from typing import List
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch
from src.config import BirdConfig, ConfigHolder
from src.model.custom_model import CustomModel
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.model.get_callbacks import get_callbacks
from src.model.get_loss import get_loss
from src.model.get_optimizer import get_optimizer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint


class CustomModelLightning(L.LightningModule):
    def __init__(self, config: BirdConfig, model: nn.Module, df):
        super().__init__()
        self.model = model
        self.validation_step_outputs = []
        self.loss = get_loss()
        self.df = df
        self.config = config 

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, y, _, x = batch

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)

        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy()
        self.log("train_accuracy", accuracy_score(y, y_hat))
        return loss

    def validation_step(self, batch, batch_idx):
        _, y, _, x = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        
        # self.validation_step_outputs.append((y, y_hat))
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy()
        self.log("val_accuracy", accuracy_score(y, y_hat))
        return loss
    
    # def on_validation_epoch_end(self):
    #     # all_preds = torch.stack(self.validation_step_outputs)
        
    #     self.validation_step_outputs.clear() 

    def configure_callbacks(self) -> List[L.Callback] | L.Callback:
        return get_callbacks(self.config)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.config.scheduler.step_size, gamma=self.config.scheduler.gamma
                ),
                "monitor": "val_loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
        }
    }
         
