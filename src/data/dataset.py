import torch
from torch.utils.data import Dataset
import librosa
import torchaudio
from torchaudio import transforms
import numpy as np
import torch.nn as nn

from src.config import BirdConfig, ConfigHolder
from src.utils.get_spectrogram_transforms import get_spectrogram_transforms
from src.utils.get_random_clip import get_rendom_clip, standardize_waveform

audio_cache = {}


class BirdClefDataset(Dataset):
    def __init__(self, df, config: BirdConfig):
        self.df = df
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.df.loc[idx, "x"])
        target = self.df.loc[idx, "y"]
        # move waveform to the device
        # waveform = waveform.to(self.config.device)
        # sample_rate = sample_rate.to(self.config.device)

        
        waveform = get_rendom_clip(self.config, waveform, sample_rate)
        waveform = standardize_waveform(self.config, waveform, sample_rate)
        waveform = waveform.reshape(
            1,
            self.config.data.sample_rate * self.config.data.frame_length,
        )
        spec = get_spectrogram_transforms(self.config)(waveform)
        assert len(spec.shape) == 3
        return (waveform, target, sample_rate, spec)
