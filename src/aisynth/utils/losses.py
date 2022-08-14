import torch


def spectral_loss(x_in, x_out, **kwargs):
    return torch.stft(x_in, x_out, **kwargs)


def norm_spec_loss(x_in, x_out, **kwargs):
    return torch.norm(spectral_loss(x_in, x_out, **kwargs), p=2, dim=-1)


def norm(x):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()


def l1_loss(x_target, x_pred, **kwargs) -> torch.Tensor:
    return torch.mean(torch.abs(x_pred - x_target), **kwargs)


