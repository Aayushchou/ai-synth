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
        log: print outputs to see training progress. TODO: add tensorboard support.
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

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def log(self, val: float) -> None:
        print(f"EPOCH LOSS: {val}")

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def fit(self) -> None:
        self.model.train()
        for _ in range(self.n_epochs):
            loss = self.train_loop(self, **self.kwargs)
            self.log(loss)
