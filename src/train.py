import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from src.data.get_data_loaders import get_data_loaders
from src.data.dataset import BirdClefDataset
from src.data.get_classified_df import get_classified_df
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from dataclasses import dataclass
import hydra
import torch.nn as nn
import torch
from lightning.pytorch.callbacks import DeviceStatsMonitor

from src.config import BirdConfig, ConfigHolder
from src.model.custom_model import CustomModel
from src.model.custom_model_lightning import CustomModelLightning
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback


def train(config: BirdConfig):
    torch.set_float32_matmul_precision(
        config.train.float32_matmul_precision
    )
    wandb_logger = WandbLogger(project="bird_clef_2024", id="exp1")
    wandb_logger.experiment.config.update({
        "batch_size": config.train.batch_size,
    }, allow_val_change=True) 

    df, train_loader, val_loader = get_data_loaders(config)
    num_classes = len(df["species"].unique())
 
    model = CustomModel(backbone_name="efficientnet_b1", num_classes=num_classes)
 
    model_wrapper = CustomModelLightning(config, model, df)

    trainer = L.Trainer(
        max_epochs=config.train.epoch_number,
        logger=wandb_logger,
        fast_dev_run=config.train.fast_dev_run
    )
    trainer.fit(
        model_wrapper,
        train_loader,
        val_dataloaders=val_loader,
        # ckpt_path=config.train.checkpoint_path,
    )
    wandb.finish()
