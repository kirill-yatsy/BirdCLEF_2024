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
    BirdConfig,
    DataProcessing,
    ModelConfig,
    SchedulerType,
    SplittedDatasets,
    TrainType,
)

FRAME_LENGTH = 5

FINE_TUNE = True

SOURCES = {
    "2024": "data/birdclef-2024/train_audio"
}

if not FINE_TUNE:
    SOURCES["2023"] = "data/birdclef-2023/train_audio"
    SOURCES["2022"] = "data/birdclef-2022/train_audio"
    SOURCES["2021"] = "data/birdclef-2021/train_short_audio"

image_width = 600
audio_len = 32000 * FRAME_LENGTH
hop_length = audio_len // (image_width - 1)

EFFICIENTNET_CONFIG = BirdConfig(
    id="1",
    seed=42,
    mode="train",
    device="cuda",
    data_processing=DataProcessing(),
    train=TrainType(
        fast_dev_run=False,
        epoch_number=20,
        batch_size=128,
        timm_model="tf_efficientnet_b0_ns",
        optimizer="adam",
        lr=0.001,
        num_workers=8,
        checkpoint_path="checkpoints/efficientnet_b0/model-fine-tune-epoch=00-train_loss=4.69.ckpt",
        float32_matmul_precision="high",
        save_model_path="checkpoints/efficientnet_b0",
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
    augmentations=torch.nn.Sequential(
        *[
            MonoToThreeChannel(
                sample_rate=32000, 
                n_mels=128,
                n_fft=2048,
                hop_length=2048, 
                top_db=100, 
                f_min=40,
                f_max=15000, 
                n_mfcc=20,
                n_chroma=12
            ),
            NormalizeData(),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=FRAME_LENGTH),
            torchaudio.transforms.TimeMasking(time_mask_param=FRAME_LENGTH),
        ]
    ),

    model=ModelConfig(in_channels=3, freeze_backbone=FINE_TUNE),
)

if __name__ == "__main__":
    print(EFFICIENTNET_CONFIG.train.float32_matmul_precision)
