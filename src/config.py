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
class DataType:
    splitted_datasets: SplittedDatasets
    short_audio_stategy: ShortAudioStategy
    sample_rate: int
    frame_length: int
    n_mels: int
    nfft: int
    hop_length: int
    fmax: int
    fmin: int
    top_db: int


@dataclass
class TrainType:
    fast_dev_run: bool
    epoch_number: int
    batch_size: int
    timm_model: str
    optimizer: str
    lr: float
    scheduler: str
    criterion: str
    num_workers: int
    pin_memory: True
    device: str
    float32_matmul_precision: str
    save_model_path: str
    save_model_name: str
    save_model_every_epoch: bool
    save_model_every_epoch_path: str
    save_model_every_epoch_name: str
    save_model_every_epoch_extension: str
    save_model_every_epoch_overwrite: bool
    save_model_every_epoch_keep_last: int
    checkpoint_path: str


@dataclass
class SchedulerType:
    step_size: int
    gamma: float


@dataclass
class BirdConfig:
    seed: int
    mode: str
    device: str
    data: DataType
    train: TrainType


class ConfigHolder:
    config: BirdConfig = None

    @staticmethod
    def set_instance(cfg: DictConfig):
        ConfigHolder.config = cfg

    @staticmethod
    def get_instance():
        return ConfigHolder.config
