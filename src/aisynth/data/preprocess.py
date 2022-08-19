import torchaudio.transforms as T
import torch


def resample(signal: torch.Tensor, sr: int, target_sr: int, **kwargs) -> torch.Tensor:
    resampler = T.Resample(sr, target_sr, **kwargs)
    signal = resampler(signal)
    return signal
