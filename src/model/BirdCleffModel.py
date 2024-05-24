import os
from typing import List
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch 
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import CONFIG
from src.configs.base_config import BirdConfig
from src.metrics.AccuracyPerClass import AccuracyPerClass
from src.model.get_callbacks import BirdCleffModelConfig, get_callbacks
from src.model.get_loss import get_loss
from src.model.get_optimizer import get_optimizer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
import torchmetrics
import timm
import wandb
import numpy as np

class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.conv(x)
        attention_map = self.sigmoid(attention_map)  # Normalize to range [0, 1]
        return x * attention_map


class Model(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
            in_chans=CONFIG.model.in_channels,
        )

        # Remove the original classification head of the backbone
        self.backbone.classifier = nn.Identity()
        if (CONFIG.model.freeze_backbone):
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.attention = SpatialAttentionModule(self.backbone.num_features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x


class BirdCleffModel(L.LightningModule):
    def __init__(self, df, num_classes):
        super().__init__()

        self.model = Model(
            backbone_name=CONFIG.train.timm_model, num_classes=num_classes
        )
        # self.validation_step_outputs = torch.tensor([])
        self.loss = get_loss()
        self.df = df
        self.num_classes = num_classes 
         
        # targetToLabelMapper = dict(enumerate(self.df["species"].unique()))

        self.init_epoch_outputs()
        # self.init_step_outputs()

    def init_epoch_outputs(self):
        self.training_epoch_outputs = {
            "y_hat": torch.tensor([]),
            "y": torch.tensor([]),
        }
        self.validation_epoch_outputs = {
            "y_hat": torch.tensor([]),
            "y": torch.tensor([]),
        }

    # def on_load_checkpoint(self, checkpoint):
    #     checkpoint["optimizer_states"] = []
    #     self.loss = nn.CrossEntropyLoss()

    # def init_step_outputs(self):
    #     self.training_current_step_outputs = {"y_hat": torch.tensor([]), "y": torch.tensor([])}
    #     self.validation_current_step_outputs = {"y_hat": torch.tensor([]), "y": torch.tensor([])}

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, y, _, x = batch
        y = y.long()
        y_hat = self(x)

        # append the y_hat and y to the training_epoch_outputs
        self.training_epoch_outputs["y_hat"] = torch.cat(
            (self.training_epoch_outputs["y_hat"], y_hat.cpu())
        )
        self.training_epoch_outputs["y"] = torch.cat(
            (self.training_epoch_outputs["y"], y.cpu())
        )

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss) 

        return {"loss": loss, "y_hat": y_hat.cpu(), "y": y.cpu()}
 

    def validation_step(self, batch, batch_idx):
        _, y, _, x = batch
        y = y.long()
        y_hat = self(x)

        # with open('y.npy', 'wb') as f:
        #     np.save(f, y.cpu().numpy()) 
        # with open('y_hat.npy', 'wb') as f:
        #     np.save(f, y_hat.cpu().numpy())
        # save y_hat and y to file
        # y_hat.cpu().numpy().tofile("y_hat.txt")
        # y.cpu().numpy().tofile("y.txt")
        
        # append the y_hat and y to the validation_step_outputs
        # concat the y_hat and y to the validation_epoch_outputs
        self.validation_epoch_outputs["y_hat"] = torch.cat(
            (self.validation_epoch_outputs["y_hat"], y_hat.cpu())
        )
        self.validation_epoch_outputs["y"] = torch.cat(
            (self.validation_epoch_outputs["y"], y.cpu())
        )

        loss = self.loss(y_hat, y)

        
        self.log("val_loss", loss)

        return {
            "loss": loss,
            "y_hat": y_hat.cpu(),
            "y": y.cpu(),
        }

    # def on_validation_batch_end(self):
    #     self.init_step_outputs()

    def on_test_epoch_start(self):
        self.init_epoch_outputs()

    def configure_callbacks(self):
        return get_callbacks()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=CONFIG.scheduler.step_size,
                    gamma=CONFIG.scheduler.gamma,
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
