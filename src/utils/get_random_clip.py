import librosa
import torch

from src.config import ConfigHolder


def standardize_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if len(waveform) > 1:
        waveform = librosa.to_mono(waveform.numpy())
        waveform = torch.tensor(waveform)
    if sample_rate != ConfigHolder.config.data.sample_rate:
        waveform = librosa.resample(
            waveform.numpy(), sample_rate, ConfigHolder.config.data.sample_rate
        )
        waveform = torch.tensor(waveform)

    return waveform


def get_rendom_clip(waveform, sample_rate, frame_size=5) -> torch.Tensor:
    """
    Get a random 5 second clip from the audio file
    :param waveform: waveform of the audio file
    :param sample_rate: sample rate of the audio file
    :param frame_size: size of the clip in seconds
    :return: 5 second clip
    """
    short_audio_stategy = ConfigHolder.config.data.short_audio_stategy

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
