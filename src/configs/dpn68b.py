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

FINE_TUNE = True


DPN68B_CONFIG = BirdConfig(
    id="1",
    seed=42,
    mode="train",
    device="cuda",
    data_processing=DataProcessing(),
    train=TrainType(
        fast_dev_run=False,
        epoch_number=100,
        batch_size=64,
        timm_model="dpn68b",
        optimizer="adam",
        lr=0.001,
        num_workers=8,
        checkpoint_path="checkpoints/dpn68b/model-epoch=35-train_loss=0.00.ckpt",
        fine_tune_checkpoint_path="checkpoints/dpn68b/model-fine-tune1-epoch=48-val_loss=-8073.44.ckpt",
        float32_matmul_precision="high",
        save_model_path="checkpoints/dpn68b",
        save_model_every_epoch_overwrite=True,
        save_model_every_epoch_keep_last=5,
        gradient_clip_val=0.5,
        fine_tune=FINE_TUNE,
    ),
    scheduler=SchedulerType(
        step_size=1,
        gamma=0.1,
    ),
    augmentations=Augmentations(
        useMixup=True,
    ),
    model=ModelConfig(in_channels=3, freeze_backbone=FINE_TUNE),
    fine_tune_path="production/dpn68b/v2",
)
