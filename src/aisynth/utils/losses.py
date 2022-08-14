import torch


def spectral_loss(x_in, x_out, **kwargs):
    return torch.stft(x_in, x_out, **kwargs)


def norm_spec_loss(x_in, x_out, **kwargs):
    return torch.norm(spectral_loss(x_in, x_out, **kwargs), p=2, dim=-1)


def norm(x):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()


def loss_fn(func_choice, x_target, x_pred):
    if func_choice == 'l1':
        return torch.mean(torch.abs(x_pred - x_target))
    elif func_choice == 'l2':
        return torch.mean((x_pred - x_target) ** 2)
    else:
        assert False, f"Unknown loss function {loss_fn}"

