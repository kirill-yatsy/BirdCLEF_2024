import librosa
import torch
import torchaudio

from src.configs.base_config import BirdConfig
 


def standardize_waveform(config: BirdConfig, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if len(waveform) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != config.data_processing.sample_rate:
        waveform = torchaudio.transforms.Resample( sample_rate, config.data_processing.sample_rate, dtype=waveform.dtype)(waveform)

    return waveform


def get_rendom_clip(config: BirdConfig, waveform, sample_rate, frame_size=15) -> torch.Tensor:
    """
    Get a random frame_size second clip from the audio file
    :param waveform: waveform of the audio file
    :param sample_rate: sample rate of the audio file
    :param frame_size: size of the clip in seconds
    :return: frame_size second clip
    """
    short_audio_stategy = config.data_processing.short_audio_stategy

    duration = len(waveform[0]) / sample_rate

    # add 1 second because second parameter of torch.randint should be more than 0
    if (duration - 1) < frame_size:
        # repeat the waveform until it reaches the frame size
        while (duration - 1) < frame_size:
            waveformToAdd = waveform
            if short_audio_stategy == "cut_first_10_percent_and_add_to_track":
                waveformToAdd = waveform[:, int(len(waveform[0]) / 10) :]

            waveform = torch.cat((waveform, waveformToAdd), dim=1)
            duration = len(waveform[0]) / sample_rate

    # return a random 5 second clip
    start = torch.randint(0, int(duration - frame_size) * sample_rate, (1,)).item()
    end = start + frame_size * sample_rate

    return waveform[:, start:end]
