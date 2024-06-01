import torch
import torchaudio
import torchaudio.transforms as T 
import librosa
import torch.nn as nn   
import random

class NoOpTransform(torch.nn.Module):
    def forward(self, waveform):
        return waveform
    
# TODO: cause memmery overflow for some reason
class RandomPitchShift(torch.nn.Module):
    def __init__(self, sample_rate, max_shift):
        super(RandomPitchShift, self).__init__()
        self.sample_rate = sample_rate
        self.max_shift = max_shift

    def forward(self, waveform):
        shift = random.uniform(-self.max_shift, self.max_shift)
        pitch_shift = torchaudio.transforms.PitchShift(self.sample_rate, shift)
        return pitch_shift(waveform)
    
class RandomGain(torch.nn.Module):
    def __init__(
        self, min_gain=-20, max_gain=20, gain_types=["db", "amplitude", "power"]
    ):
        super(RandomGain, self).__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.gain_types = gain_types

    def get_min_max_for_amplitude(self):
        return 10 ** (self.min_gain / 20), 10 ** (self.max_gain / 20)

    def get_min_max_for_power(self):
        return 10 ** (self.min_gain / 10), 10 ** (self.max_gain / 10)

    def forward(self, waveform):
        gain_type = random.choice(self.gain_types)
        if gain_type == "db":
            min_value, max_value = self.min_gain, self.max_gain
        elif gain_type == "amplitude":
            min_value, max_value = self.get_min_max_for_amplitude()
        else:
            min_value, max_value = self.get_min_max_for_power()

        gain = random.uniform(min_value, max_value)
        # print(f"Gain: {gain} {gain_type}")
        gain = torchaudio.transforms.Vol(gain, gain_type=gain_type)
        return gain(waveform)
    
class RandomGainTransition(torch.nn.Module):
    def __init__(self, min_gain_db=-20, max_gain_db=20, duration=None):
        super(RandomGainTransition, self).__init__()
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.duration = duration

    def forward(self, waveform):
        start_gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        end_gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        if self.duration is None:
            self.duration = waveform.shape[1]
        gains = torch.linspace(start_gain_db, end_gain_db, self.duration)
        waveform = waveform * torch.pow(10.0, gains / 20.0)
        return waveform
    
class RandomGaussianNoise(torch.nn.Module):
    def __init__(self, min_std=0.01, max_std=0.1):
        super(RandomGaussianNoise, self).__init__()
        self.min_std = min_std
        self.max_std = max_std

    def forward(self, waveform):
        std = random.uniform(self.min_std, self.max_std)
        noise = torch.randn_like(waveform) * std
        return waveform + noise
    
class RandomGaussianSNR(torch.nn.Module):
    def __init__(self, min_snr_db=10, max_snr_db=40):
        super(RandomGaussianSNR, self).__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def forward(self, waveform):
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise
    
# class RandomAddShortNoises(torch.nn.Module):
#     def __init__(self, noise_files, sample_rate, min_snr_db=10, max_snr_db=30, p=0.5):
#         super(RandomAddShortNoises, self).__init__()
#         self.noise_files = noise_files
#         self.sample_rate = sample_rate
#         self.min_snr_db = min_snr_db
#         self.max_snr_db = max_snr_db
#         self.p = p

#     def forward(self, waveform):
#         if random.random() < self.p:
#             noise_file = random.choice(self.noise_files)
#             noise, _ = torchaudio.load(noise_file)
#             noise = torchaudio.transforms.Resample(orig_freq=noise.shape[1], new_freq=self.sample_rate)(noise)
#             noise = noise[:, :waveform.shape[1]]
#             snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
#             signal_power = waveform.pow(2).mean()
#             noise_power = noise.pow(2).mean()
#             scale_factor = torch.sqrt(signal_power / noise_power / (10 ** (snr_db / 10)))
#             return waveform + scale_factor * noise
#         return waveform
    
class RandomLowPassFilter(torch.nn.Module):
    def __init__(self, min_cutoff_freq=1000, max_cutoff_freq=6000, sample_rate=32000):
        super(RandomLowPassFilter, self).__init__()
        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        self.sample_rate = sample_rate

    def forward(self, waveform):
        cutoff_freq = random.uniform(self.min_cutoff_freq, self.max_cutoff_freq)
        return torchaudio.functional.lowpass_biquad(waveform, self.sample_rate, cutoff_freq)
    
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