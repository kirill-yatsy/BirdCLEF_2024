import torch
import yaml 
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

 

EFFICIENTNET_CONFIG_B2= BirdConfig(
    id="1",
    seed=42,
    mode="train",
    device="cuda",
    data_processing=DataProcessing(),
    train=TrainType(
        fast_dev_run=False,
        epoch_number=70,
        batch_size=64,
        timm_model="efficientnet_b2",
        optimizer="adam",
        lr=0.001,
        num_workers=8,
        checkpoint_path="checkpoints/efficientnet_b2/model-epoch=10-train_loss=0.07.ckpt",
        fine_tune_checkpoint_path=None,
        float32_matmul_precision="high",
        save_model_path="checkpoints/efficientnet_b2",
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
    fine_tune_path="production/efficientnet_b2/v1",
)
 