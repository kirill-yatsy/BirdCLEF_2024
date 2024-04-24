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


@dataclass
class TrainType:
    epoch_number: int
    batch_size: int


@dataclass
class BirdConfig:
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
