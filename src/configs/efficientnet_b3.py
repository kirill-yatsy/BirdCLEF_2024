import torch
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
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

FRAME_LENGTH = 15

FINE_TUNE = False

SOURCES = {
    "2024": "data/birdclef-2024/train_audio"
}

if not FINE_TUNE:
    SOURCES["2023"] = "data/birdclef-2023/train_audio"
    SOURCES["2022"] = "data/birdclef-2022/train_audio"
    SOURCES["2021"] = "data/birdclef-2021/train_short_audio"
    SOURCES["extended"] = "data/xeno-canto-extended"
    

EFFICIENTNET_CONFIG_B3= BirdConfig(
    id="1",
    seed=42,
    mode="train",
    device="cuda",
    data_processing=DataProcessing(),
    train=TrainType(
        fast_dev_run=False,
        epoch_number=20,
        batch_size=64,
        timm_model="efficientnet_b3",
        optimizer="adam",
        lr=0.001,
        num_workers=8,
        checkpoint_path="checkpoints/efficientnet_b3/model-fine-tune-epoch=02-train_loss=6.83.ckpt",
        float32_matmul_precision="high",
        save_model_path="checkpoints/efficientnet_b3",
        save_model_every_epoch_overwrite=True,
        save_model_every_epoch_keep_last=5, 
        gradient_clip_val=0.5,
        fine_tune=FINE_TUNE,
    ),
    scheduler=SchedulerType(
        step_size=1,
        gamma=0.1,
    ),
    datasets=SplittedDatasets(
        birdclefs=SOURCES
    ),
    augmentations=Augmentations(
        useMixup=False,
    ),

    model=ModelConfig(in_channels=3, freeze_backbone=FINE_TUNE),

)
 