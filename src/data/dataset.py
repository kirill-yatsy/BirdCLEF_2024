import torch
from torch.utils.data import Dataset
import librosa
import torchaudio
from torchaudio import transforms
import numpy as np
import torch.nn as nn

from src.config import ConfigHolder
from src.data.get_spectrogram_transforms import get_spectrogram_transforms
from src.utils.get_random_clip import get_rendom_clip, standardize_waveform

audio_cache = {}


class BirdClefDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.df.loc[idx, "x"])
        
        # move waveform to the device
        # waveform = waveform.to(ConfigHolder.config.device)
        # sample_rate = sample_rate.to(ConfigHolder.config.device)
        
        target = self.df.loc[idx, "y"]
        waveform = get_rendom_clip(waveform, sample_rate)
        waveform = standardize_waveform(waveform, sample_rate)
        waveform = waveform.reshape(
            1,
            ConfigHolder.config.data.sample_rate
            * ConfigHolder.config.data.frame_length,
        )
        spec = get_spectrogram_transforms()(waveform)
        return (waveform, target, sample_rate, spec)
