import torch
import torchaudio
import torchaudio.transforms as T 
import librosa
import torch.nn as nn  
from src.configs.base_config import BirdConfig
def generate_mel_spectrogram(waveform, sample_rate, n_mels, n_fft, hop_length, f_min, f_max, top_db):
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
    )
    amplitude_db_transform = torchaudio.transforms.AmplitudeToDB(top_db=top_db)
    
    mel_spectrogram = mel_spectrogram_transform(waveform)
    mel_spectrogram_db = amplitude_db_transform(mel_spectrogram)
    
    return mel_spectrogram_db

def generate_mfcc(waveform, sample_rate, n_mfcc, n_mels, n_fft, hop_length, f_min, f_max):
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "f_min": f_min,
            "f_max": f_max,
        },
    )
    mfcc = mfcc_transform(waveform)
    return mfcc

def generate_chroma_feature(waveform, sr, n_fft, hop_length, n_chroma, epsilon=1e-6):
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
    )
    magnitude = stft.abs() + epsilon  # Adding epsilon to avoid log(0) issues
    chroma_filter = librosa.filters.chroma(sr=sr, n_fft=n_fft, n_chroma=n_chroma)
    chroma_filter = torch.tensor(chroma_filter, dtype=torch.float32)
    chroma = torch.matmul(chroma_filter, magnitude.squeeze(0))
    chroma = chroma / torch.max(chroma) + epsilon
    return chroma

class MonoToThreeChannel(nn.Module):
    def __init__(self, sample_rate, n_mels, n_fft, hop_length, f_min, f_max, top_db, n_mfcc, n_chroma):
        super(MonoToThreeChannel, self).__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.top_db = top_db
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma

    def forward(self, waveform):
        # Generate Mel Spectrogram
        mel_spectrogram_db = generate_mel_spectrogram(
            waveform, 
            self.sample_rate, 
            self.n_mels, 
            self.n_fft, 
            self.hop_length, 
            self.f_min, 
            self.f_max, 
            self.top_db
        )

        # Generate MFCC
        mfcc = generate_mfcc(
            waveform, 
            self.sample_rate, 
            self.n_mfcc, 
            self.n_mels, 
            self.n_fft, 
            self.hop_length, 
            self.f_min, 
            self.f_max
        )

        # Generate Chroma Features
        chroma = generate_chroma_feature(
            waveform, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_chroma=self.n_chroma
        ).unsqueeze(0)

        # Resize MFCC and Chroma to match Mel Spectrogram dimensions
        mfcc_resized = nn.functional.interpolate(
            mfcc.unsqueeze(0), size=mel_spectrogram_db.shape[1:], mode="bilinear"
        ).squeeze(0)
        chroma_resized = nn.functional.interpolate(
            chroma.unsqueeze(0), size=mel_spectrogram_db.shape[1:], mode="bilinear"
        ).squeeze(0)

        # Stack to create a 3-channel image
        final_output = torch.stack([mel_spectrogram_db, mfcc_resized, chroma_resized], dim=0).squeeze(1)
        return final_output
    
class NormalizeData(torch.nn.Module):
    def __init__(self):
        super(NormalizeData, self).__init__()

    def forward(self, x):
        min_val = torch.min(x)
        max_val = torch.max(x)
        if max_val - min_val == 0:
            return x
        return (x - min_val) / (max_val - min_val)
    

class MixUpAugmentation(nn.Module):
    def __init__(self, alpha=1.0):
        super(MixUpAugmentation, self).__init__()
        self.alpha = alpha

    def forward(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam