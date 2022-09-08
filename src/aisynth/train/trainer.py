import aisynth.globals as G

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch import Tensor
from typing import Callable


class Trainer(nn.Module):
    """Orchestrates the training of the model. TODO: add distributed training.
    params:
        model: The autoencoder model to be trained.
        train_dataset: The audio train dataset that is to be used for training.
        device: The device where the training should occur.
        optimizer: The optimizer that should be used.
        n_epochs: Number of epochs for which to train the model.
        epoch_func: The function for running the training loop for each epoch
        criterion: The loss function for calculating the loss between outputs
        batch_size: The batch size for training
        num_workers: The number of parallel workers for the dataloader
        shuffle: Whether to shuffle the input data.

    functions:
        forward: run the trained model on an input sample
        log: print outputs to see training progress.
        save: save/checkpoint the model to a given location.
        fit: run training, fit the model based on the params provided.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        device: str,
        optimizer: Optimizer,
        n_epochs: int,
        train_loop: Callable[[nn.Module, ...], float],
        criterion: Callable[[Tensor, Tensor, ...], Tensor],
        batch_size: int = 8,
        num_workers: int = 2,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.model = model.to(device)
        self.trainloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
        self.device = device
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.train_loop = train_loop
        self.kwargs = kwargs
        self.criterion = criterion
        self.writer = G.writer

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def log(self, val: float or torch.Tensor or nn.Module, label: str, log_type: str, step=None) -> None:
        if log_type == "scalar":
            self.writer.add_scalar(label, val, step)
        elif log_type == "embedding":
            self.writer.add_embedding(val, global_step=step)
        elif log_type == "audio":
            self.writer.add_audio(label, val)
        elif log_type == "model":
            self.writer.add_graph(self.model, next(iter(self.trainloader))[0])
        else:
            print("please select a valid log type: (scalar, embedding, audio, model)")

    def save_checkpoint(self, path: str) -> None:

        with torch.no_grad():
            torch.save({'model': self.model.state_dict(),  # should also save bottleneck k's as buffers
                        'opt': self.optimizer.state_dict() if self.optimizer is not None else None,
                        **self.kwargs}, path)

    def fit(self) -> None:
        self.model.train()
        for i in range(self.n_epochs):
            loss = self.train_loop(self, **self.kwargs)
            self.log(loss, "Epoch Loss", "scalar", step=i)
        self.writer.flush()
        self.writer.close()
