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
from src.model.BirdCleffModel import BirdCleffModel
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L 
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import NeptuneLogger
import neptune

def train(config: BirdConfig):
    torch.set_float32_matmul_precision(config.train.float32_matmul_precision)
    # wandb_logger = WandbLogger(project="bird_clef_2024", id=config.id)
    neptune_logger = NeptuneLogger(
        project="kirill.yatsy/birdclef-2024",
        name=config.id,
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2Zjc0N2I0NC02Mjk4LTQ5ZDYtODljMy0yZGMwNTYzZDNhODcifQ==",
        log_model_checkpoints=False, 
    )

    df, train_loader, val_loader = get_data_loaders(config)
    num_classes = len(df["species"].unique())

    model_wrapper = BirdCleffModel(config, df, num_classes)

    trainer = L.Trainer(
        max_epochs=config.train.epoch_number,
        logger=neptune_logger,
        fast_dev_run=config.train.fast_dev_run,
        gradient_clip_val=0.5,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
    )
    trainer.fit(
        model_wrapper,
        train_loader,
        val_dataloaders=val_loader,
        # ckpt_path=config.train.checkpoint_path if config.train.checkpoint_path else None,
    )
