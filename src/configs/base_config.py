import torch
import torchaudio
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from dataclasses import dataclass
import enum
from torchvision.transforms import v2
from src.augmentations.get_spectrogram_augmentations import MonoToThreeChannel, NoOpTransform, NormalizeData, RandomGain, RandomGainTransition, RandomGaussianNoise, RandomGaussianSNR, RandomLowPassFilter, RandomPitchShift

FRAME_LENGTH = 20
image_width = 600
audio_len = 32000 * FRAME_LENGTH
hop_length = audio_len // (image_width - 1)

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
    hop_length: int = hop_length
    fmax: int = 16000
    fmin: int = 0
    top_db: int = 80
    csv_path: str = "data/processed/train_df.csv"
    fine_tune_csv_path: str = "data/processed/fine_tune_df.csv"

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
class Augmentations:
    useMixup: bool = True
    mixup_alpha: float = 1

    sequence: torch.nn.Sequential = v2.Compose([ 
        v2.RandomChoice([RandomGain(), RandomGainTransition()]),
        v2.RandomChoice([RandomGaussianNoise(), RandomGaussianSNR()]),
        v2.RandomChoice([RandomLowPassFilter(), NoOpTransform()]), 
        MonoToThreeChannel(
            sample_rate=32000, 
            n_mels=128,
            n_fft=2048,
            hop_length=512, 
            top_db=80, 
            f_min=0,
            f_max=16000, 
            n_mfcc=20,
            n_chroma=12
        ),
        NormalizeData(),
        v2.RandomChoice([torchaudio.transforms.FrequencyMasking(freq_mask_param=24, iid_masks=True), NoOpTransform()]),
        v2.RandomChoice([torchaudio.transforms.TimeMasking(time_mask_param=60, iid_masks=True, p=0.5), NoOpTransform()]),
    
    ])

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
    augmentations: Augmentations
    model: ModelConfig 

