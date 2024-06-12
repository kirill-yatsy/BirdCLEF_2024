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
 

FINE_TUNE = False
 
SERESNEXT26T_32X4D_CONFIG = BirdConfig(
    id="1",
    seed=42,
    mode="train",
    device="cuda",
    data_processing=DataProcessing(),
    train=TrainType(
        fast_dev_run=False,
        epoch_number=50,
        batch_size=128,
        timm_model="seresnext26tn_32x4d",
        optimizer="adam",
        lr=0.001,
        num_workers=12,
        checkpoint_path="checkpoints/seresnext26tn_32x4d/model-epoch=09-train_loss=0.07.ckpt",
        float32_matmul_precision="high",
        save_model_path="checkpoints/seresnext26tn_32x4d",
        save_model_every_epoch_overwrite=True,
        save_model_every_epoch_keep_last=5, 
        gradient_clip_val=0.4,
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

)
 