import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from dataclasses import dataclass
import enum

from src.configs.base_config import BirdConfig, DataProcessing, SchedulerType, SplittedDatasets, TrainType

EFFICIENTNET_CONFIG = BirdConfig(
    id="1",
    seed=42,
    mode="train",
    device="cuda",

    data_processing=DataProcessing(
        sample_rate=32000,
        frame_length=5,
        n_mels=128,
        nfft=2048,
        hop_length=512,
        fmax=16000,
        fmin=0,
        top_db=80,
        short_audio_stategy="cut_first_10_percent_and_add_to_track",
    ),

    train=TrainType(
        fast_dev_run=False,
        epoch_number=20,
        batch_size=64,
        timm_model="tf_efficientnet_b0_ns",
        optimizer="adam",
        lr=0.001,
        num_workers=8,
        float32_matmul_precision="high",
        save_model_path="checkpoints/efficientnet_b1", 
        save_model_every_epoch_overwrite=True,
        save_model_every_epoch_keep_last=5,
        checkpoint_path=None,
        gradient_clip_val=0.5,
    ),

    scheduler=SchedulerType(
        step_size=1,
        gamma=0.1,
    ),

    datasets=SplittedDatasets(
        birdclefs={
            "2024": "data/birdclef-2024/train_audio",
            "2023": "data/birdclef-2023/train_audio",
            "2022": "data/birdclef-2022/train_audio",
            "2021": "data/birdclef-2021/train_short_audio" 
        }
    
    )
        
    
)

if __name__ == "__main__":
    print(EFFICIENTNET_CONFIG.train.float32_matmul_precision)
