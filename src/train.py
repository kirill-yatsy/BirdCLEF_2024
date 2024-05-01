import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from src.data.get_data_loaders import get_data_loaders
from src.data.StratifiedSampler import StratifiedSampler
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

from src.config import ConfigHolder
from src.model.custom_model import CustomModel
from src.model.custom_model_wrapper import CustomModelWrapper
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint


def train():
    torch.set_float32_matmul_precision(
        ConfigHolder.config.train.float32_matmul_precision
    )
    wandb_logger = WandbLogger(project="bird_clef_2024", id="exp1")
    wandb_logger.experiment.config["batch_size"] = ConfigHolder.config.train.batch_size 

    df, train_loader, val_loader = get_data_loaders()
    num_classes = len(df["species"].unique())
    # if ConfigHolder.config.train.checkpoint_path is not None:
    #     model_wrapper = CustomModelWrapper.load_from_checkpoint(
    #         ConfigHolder.config.train.checkpoint_path
    #     )

    model = CustomModel(backbone_name="efficientnet_b1", num_classes=num_classes)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_wrapper = CustomModelWrapper(model, loss, optimizer)

    trainer = L.Trainer(
        callbacks=[
            # DeviceStatsMonitor(),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(),
        ],
        max_epochs=ConfigHolder.config.train.epoch_number,
        # limit_test_batches=0.1,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        logger=wandb_logger,
        fast_dev_run=ConfigHolder.config.train.fast_dev_run,
        enable_checkpointing=ConfigHolder.config.train.save_model_every_epoch,
        default_root_dir=ConfigHolder.config.train.save_model_every_epoch_path,
    )
    trainer.fit(
        model_wrapper,
        train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ConfigHolder.config.train.checkpoint_path,
    )
    wandb.finish()
    # epoch_num = ConfigHolder.config.train.epoch_number
    # for epoch in range(epoch_num):
    #     with tqdm(train_loader, unit="batch") as tepoch:
    #         for waveforms, targets, _ in tepoch:
    #             tepoch.set_description(f"Epoch {epoch}")
    #             tepoch.set_postfix(loss=0.0)
    #             tepoch.update()

    # model.train()
    # optimizer.zero_grad()
    # outputs = model(waveforms)
    # loss = criterion(outputs, targets)
    # loss.backward()
    # optimizer.step()
    # tepoch.set_postfix(loss=loss.item())

    # model.eval()
    # outputs = model(waveforms)
    # loss = criterion(outputs, targets)
    # tepoch.set_postfix(loss=loss.item())

    print("Training the model...")
