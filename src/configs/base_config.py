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


@dataclass
class DataProcessing:
    sample_rate: int
    frame_length: int
    n_mels: int
    nfft: int
    hop_length: int
    fmax: int
    fmin: int
    top_db: int

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
    checkpoint_path: str
    gradient_clip_val: float


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