import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from src.data.get_data_loaders import get_data_loaders
from src.data.dataset import BirdClefDataset
from src.data.get_classified_df import get_classified_df
from torch.utils.data import DataLoader 
from dataclasses import dataclass
 
import torch.nn as nn
import torch
from lightning.pytorch.callbacks import DeviceStatsMonitor

from src.config import CONFIG 
from src.model.BirdCleffModel import BirdCleffModel
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L 
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import NeptuneLogger
 
from src.utils.get_backbone_weights_from_checkpoint import get_backbone_weights_from_checkpoint

def train():
    torch.set_float32_matmul_precision(CONFIG.train.float32_matmul_precision)
    neptune_logger = NeptuneLogger(
        project="kirill.yatsy/birdclef-2024",
        name=CONFIG.id,
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2Zjc0N2I0NC02Mjk4LTQ5ZDYtODljMy0yZGMwNTYzZDNhODcifQ==",
        log_model_checkpoints=False, 
        
    )

    df, train_loader, val_loader = get_data_loaders(CONFIG)
    num_classes = len(df["species"].unique())
    print(f"Number of classes: {num_classes}")

    model_wrapper = BirdCleffModel(df, num_classes)

    # filtered_state_dict = get_backbone_weights_from_checkpoint(model_wrapper.model, CONFIG.train.checkpoint_path)
    # # print(filtered_state_dict)
    # print("Loading the backbone weights")
    # model_wrapper.model.load_state_dict(filtered_state_dict, strict=False)
    # print("list of names of loaded layouts")
    # print(model_wrapper.model.state_dict().keys())

    

    trainer = L.Trainer( 
        max_epochs=CONFIG.train.epoch_number,
        logger=neptune_logger,
        # logger=False,
        fast_dev_run=CONFIG.train.fast_dev_run,
        # fast_dev_run=True,
        gradient_clip_val=CONFIG.train.gradient_clip_val, 
        # min_epochs=100,
        # accelerator="cpu",
        # max_steps=5, 
    )
    trainer.fit(
        model_wrapper,
        train_loader,
        val_dataloaders=val_loader,
        ckpt_path=CONFIG.train.checkpoint_path,
        
    )


if __name__ == "__main__":
    train()
