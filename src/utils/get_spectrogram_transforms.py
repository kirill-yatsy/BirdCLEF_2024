import torch
import torchaudio
import torchaudio.transforms as T
from src.config import BirdConfig, ConfigHolder
import librosa


def get_spectrogram_augmentations():
    return [
        # torchaudio.transforms.FrequencyMasking(freq_mask_param=config.dataset.freq_mask_param),
        # torchaudio.transforms.TimeMasking(time_mask_param=config.dataset.time_mask_param),
    ]


def generate_chroma_feature(waveform, sr, n_fft, hop_length, n_chroma, epsilon=1e-6):

    # Compute STFT
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
    )
    magnitude = stft.abs() + epsilon  # Adding epsilon to avoid log(0) issues

    # Create a chroma filter bank
    chroma_filter = librosa.filters.chroma(sr=sr, n_fft=n_fft, n_chroma=n_chroma)
    chroma_filter = torch.tensor(chroma_filter, dtype=torch.float32)

    # Apply the chroma filter bank
    chroma = torch.matmul(chroma_filter, magnitude.squeeze(0))

    # Normalize the chroma features
    chroma = chroma / torch.max(chroma) + epsilon

    return chroma


class MonoToThreeChannel(torch.nn.Module):

    def __init__(self, config: BirdConfig):
        super(MonoToThreeChannel, self).__init__()
        self.config = config

        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            config.data.sample_rate,
            n_mels=config.data.n_mels,
            n_fft=config.data.nfft,
            hop_length=config.data.hop_length,
            f_max=config.data.fmax,
            f_min=config.data.fmin,
        )
        self.amplitude_db_transform = torchaudio.transforms.AmplitudeToDB(
            top_db=config.data.top_db
        )
        self.mfcc_transform = T.MFCC(
            sample_rate=config.data.sample_rate,
            n_mfcc=config.data.n_mels,
            melkwargs={
                "n_mels": config.data.n_mels,
                "n_fft": config.data.nfft,
                "hop_length": config.data.hop_length,
                "f_min": config.data.fmin,
                "f_max": config.data.fmax,
            },
        )

    def forward(self, waveform):
        # Mel Spectrogram
        mel_spectrogram = self.mel_spectrogram_transform(waveform)
        mel_spectrogram_db = self.amplitude_db_transform(mel_spectrogram)

        # MFCC
        mfcc = self.mfcc_transform(waveform)

        # Chromagram from Mel Spectrogram
        chroma = generate_chroma_feature(
            waveform,
            sr=self.config.data.sample_rate,
            n_fft=self.config.data.nfft,
            hop_length=self.config.data.hop_length,
            n_chroma=self.config.data.n_mels,
        ).unsqueeze(0)

        # Normalize features to the same scale
        # mel_spectrogram_db = (mel_spectrogram_db - torch.min(mel_spectrogram_db)) / (
        #     torch.max(mel_spectrogram_db) - torch.min(mel_spectrogram_db)
        # )
        # mfcc = (mfcc - torch.min(mfcc)) / (torch.max(mfcc) - torch.min(mfcc))
        # chroma = (chroma - torch.min(chroma)) / (torch.max(chroma) - torch.min(chroma))

        # Resize MFCC and Chroma to match Mel Spectrogram dimensions
        mfcc_resized = torch.nn.functional.interpolate(
            mfcc.unsqueeze(0), size=mel_spectrogram_db.shape[1:], mode="bilinear"
        ).squeeze(0)
        chroma_resized = torch.nn.functional.interpolate(
            chroma.unsqueeze(0), size=mel_spectrogram_db.shape[1:], mode="bilinear"
        ).squeeze(0)

        # Stack to create a 3-channel image
        return torch.stack([mel_spectrogram_db, mfcc_resized, chroma_resized], dim=0).squeeze(1)

class NormalizeData(torch.nn.Module):
    def __init__(self):
        super(NormalizeData, self).__init__()

    def forward(self, x):
        min_val = torch.min(x)
        max_val = torch.max(x)
        if max_val - min_val == 0:
            return x
        return (x - min_val) / (max_val - min_val)

def get_spectrogram_transforms(config: BirdConfig):
    spectrogram_transforms = [
        MonoToThreeChannel(config),
        NormalizeData()
    ]

 

    if config.mode == "train":
        spectrogram_augmentations = get_spectrogram_augmentations()
        spectrogram_transforms.extend(spectrogram_augmentations)

    return torch.nn.Sequential(*spectrogram_transforms)
