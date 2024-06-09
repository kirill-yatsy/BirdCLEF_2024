import os
from typing import List
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchvision 
import lightning as L
import torch 
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import CONFIG
from src.configs.base_config import BirdConfig
# from src.metrics.AccuracyPerClass import AccuracyPerClass
from src.model.get_callbacks import BirdCleffModelConfig, get_callbacks
from src.model.get_loss import get_loss
from src.model.get_optimizer import get_optimizer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint 
import timm 
import numpy as np
from torchvision.transforms import v2

import torch.nn.functional as F

class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0 )
        # self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.conv1(x)
        # attention_map = F.relu(attention_map)
        # attention_map = self.conv2(attention_map)
        attention_map = self.sigmoid(attention_map) 
        return x * attention_map

# Learn to choose between Max Pooling and Average Pooling
class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return  torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(2, 3), keepdim=True).pow(1.0 / self.p) 

class Head(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.batchnorm1 = nn.BatchNorm1d(in_features)  # Batch normalization after first FC layer
        # self.batchnorm2 = nn.BatchNorm1d(num_classes)  # Batch normalization after second FC layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)  # Batch normalization
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

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
        self.pool = GeMPooling()
        self.head = Head(self.backbone.num_features, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def reinintialize_head(self, num_classes):
        self.head = Head(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.head(x)
        # x = self.softmax(x)
        
        return x


class BirdCleffModel(L.LightningModule):
    def __init__(self, df, num_classes):
        super().__init__()

        self.model = Model(
            backbone_name=CONFIG.train.timm_model, num_classes=num_classes
        )
        self.loss = get_loss()
        self.df = df 
 

        self.init_epoch_outputs()
        self.init_new_num_classes(num_classes)

    def init_new_num_classes(self, num_classes):
        self.mixup = v2.MixUp(num_classes=num_classes, alpha=1.0)
        self.cutmix = v2.CutMix(num_classes=num_classes)
        self.model.reinintialize_head(num_classes)
        self.num_classes = num_classes
        
 
    def init_epoch_outputs(self):
        self.training_epoch_outputs = {
            "y_hat": torch.tensor([]),
            "y": torch.tensor([]),
        }
        self.validation_epoch_outputs = {
            "y_hat": torch.tensor([]),
            "y": torch.tensor([]),
        }
    
 
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, y, _, x = batch
        y = y.long()
        if CONFIG.augmentations.useMixup:
            cutmix_or_mixup = v2.RandomChoice([self.cutmix, self.mixup])
            x,y = cutmix_or_mixup(x, y)
        

        y_hat = self(x)
        # y = y.long()
        y_hat = y_hat.float()
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if CONFIG.augmentations.useMixup:
            y = y.argmax(dim=1)
        # append the y_hat and y to the training_epoch_outputs
        self.training_epoch_outputs["y_hat"] = torch.cat(
            (self.training_epoch_outputs["y_hat"], y_hat.cpu())
        )
        self.training_epoch_outputs["y"] = torch.cat(
            (self.training_epoch_outputs["y"], y.cpu())
        )

 

        return {"loss": loss, "y_hat": y_hat.cpu(), "y": y.cpu()}
 

    def validation_step(self, batch, batch_idx):
        _, y, _, x = batch 
        y = y.float()
        y_hat = self(x) 
        y_hat = y_hat 
        
        
        self.validation_epoch_outputs["y_hat"] = torch.cat(
            (self.validation_epoch_outputs["y_hat"], y_hat.cpu())
        )
        self.validation_epoch_outputs["y"] = torch.cat(
            (self.validation_epoch_outputs["y"], y.cpu())
        )

        loss = self.loss(y_hat.argmax(dim=1).float(), y.float())

        
        self.log("val_loss", loss, on_step=True, on_epoch=True )

        return {
            "loss": loss,
            "y_hat": y_hat.cpu(),
            "y": y.cpu(),
        }
 

    def on_test_epoch_start(self):
        self.init_epoch_outputs()

    def configure_callbacks(self):
        return get_callbacks()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=CONFIG.train.epoch_number ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
