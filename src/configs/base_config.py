import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from dataclasses import dataclass
import enum


class ShortAudioStategy(enum.Enum):
    MERGE_CUTTED = "cut_first_10_percent_and_add_to_track"


@dataclass
class SplittedDatasets:
    birdclefs: dict[str, str]
    # add optional birdclefs_fine_tune field
    birdclefs_fine_tune: dict[str, str] = None

@dataclass
class ModelConfig:
    in_channels: int
    freeze_backbone: bool

 
@dataclass
class DataProcessing:
    sample_rate: int = 32000
    frame_length: int = 10
    n_mels: int = 128
    nfft: int = 2048
    hop_length: int = 512
    fmax: int = 16000
    fmin: int = 0
    top_db: int = 80
    csv_path: str = "data/processed/train_df.csv"

    short_audio_stategy: enum.Enum = "cut_first_10_percent_and_add_to_track"

 


@dataclass
class TrainType:
    fast_dev_run: bool
    epoch_number: int
    batch_size: int
    timm_model: str
    optimizer: str
    lr: float
    num_workers: int
    float32_matmul_precision: str
    save_model_path: str 
    save_model_every_epoch_overwrite: bool
    save_model_every_epoch_keep_last: int 
    gradient_clip_val: float
    fine_tune: bool
    checkpoint_path: str = None


@dataclass
class SchedulerType:
    step_size: int
    gamma: float


@dataclass
class BirdConfig:
    id: str
    seed: int
    mode: str
    device: str
    data_processing: DataProcessing
    train: TrainType
    scheduler: SchedulerType
    datasets: SplittedDatasets
    augmentations: list[any]
    model: ModelConfig