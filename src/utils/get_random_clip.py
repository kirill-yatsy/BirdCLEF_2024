import torch

from src.utils.get_config import get_config


def get_rendom_clip(waveform, sample_rate, frame_size=5) -> torch.Tensor:
    """
    Get a random 5 second clip from the audio file
    :param waveform: waveform of the audio file
    :param sample_rate: sample rate of the audio file
    :param frame_size: size of the clip in seconds
    :return: 5 second clip
    """
    config = get_config()
    short_audio_stategy = config["data"]["short_audio_stategy"]

    duration = len(waveform[0]) / sample_rate
    if duration < frame_size:
        # repeat the waveform until it reaches the frame size
        while duration < frame_size:
            waveformToAdd = waveform
            if short_audio_stategy == "cut_first_10_percent_and_add_to_track":
                waveformToAdd = waveform[:, int(len(waveform[0]) / 10) :]
            
            waveform = torch.cat((waveform, waveformToAdd), dim=1)
            duration = len(waveform[0]) / sample_rate
        return waveform

    start = torch.randint(0, int(duration - frame_size) * sample_rate, (1,)).item()
    return waveform[:, start : start + frame_size * sample_rate]
