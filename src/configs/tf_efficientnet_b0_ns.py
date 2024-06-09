import torch
import yaml
 
from dataclasses import dataclass
import enum
import torchaudio
from src.augmentations.get_spectrogram_augmentations import (
    MonoToThreeChannel,
    NormalizeData,
)
from src.configs.base_config import (
    Augmentations,
    BirdConfig,
    DataProcessing,
    ModelConfig,
    SchedulerType,
    SplittedDatasets,
    TrainType,
)
from torchvision.transforms import v2

FINE_TUNE = True

EFFICIENTNET_CONFIG = BirdConfig(
    id="1",
    seed=42,
    mode="train",
    device="cuda",
    data_processing=DataProcessing(),
    train=TrainType(
        fast_dev_run=False,
        epoch_number=6,
        batch_size=128,
        timm_model="tf_efficientnet_b0_ns",
        optimizer="adam",
        lr=0.001,
        num_workers=10,
        checkpoint_path="checkpoints/tf_efficientnet_b0_ns/model-epoch=32-train_loss=0.00.ckpt",
        float32_matmul_precision="high",
        save_model_path="checkpoints/tf_efficientnet_b0_ns",
        save_model_every_epoch_overwrite=True,
        save_model_every_epoch_keep_last=-1,
        gradient_clip_val=0.5,
        fine_tune="checkpoints/tf_efficientnet_b0_ns/model-epoch=45-train_loss=0.00.ckpt",
    ),
    scheduler=SchedulerType(
        step_size=1,
        gamma=0.1,
    ), 
    augmentations=Augmentations(
        useMixup=True, 
    ),
    model=ModelConfig(in_channels=3, freeze_backbone=FINE_TUNE),

    fine_tune_path="production/tf_efficientnet_b0_ns/v1",
)
 