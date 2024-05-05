import torch
import torchaudio

from src.config import BirdConfig, ConfigHolder


def get_spectrogram_augmentations():
    return [
        # torchaudio.transforms.FrequencyMasking(freq_mask_param=config.dataset.freq_mask_param),
        # torchaudio.transforms.TimeMasking(time_mask_param=config.dataset.time_mask_param),
    ]


def get_spectrogram_transforms(config: BirdConfig):
    spectrogram_transforms = []

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        config.data.sample_rate,
        n_mels=config.data.n_mels,
        n_fft=config.data.nfft,
        hop_length=config.data.hop_length,
        f_max=config.data.fmax,
        f_min=config.data.fmin,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
        top_db=config.data.top_db
    )

    spectrogram_transforms.append(mel_spectrogram)
    spectrogram_transforms.append(amplitude_to_db)

    if config.mode == "train":
        spectrogram_augmentations = get_spectrogram_augmentations()
        spectrogram_transforms.extend(spectrogram_augmentations)

    return torch.nn.Sequential(*spectrogram_transforms)
