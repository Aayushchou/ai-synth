import torch.nn as nn
import aisynth.globals as G


def simple_train_loop(trainer: nn.Module, **kwargs) -> float:
    """Function for running a simple training loop for a single epoch.
    params:
        trainer: aisynth.train.Trainer object, with all the training parameters.
            including: optimizer, model, trainloader, device, criterion
        **kwargs: additional arguments for the loss function
    procedure:
        1. Input is fed into the model.
        2. Output is compared to the input.
        3. Loss is produced.
        4. Back-propagation
        5. Total loss is calculated.
    returns:
        total loss for the epoch"""
    running_loss = 0.0
    for audio_in, sr in trainer.trainloader:
        audio_in, sr = audio_in.to(trainer.device), sr.to(trainer.device)
        trainer.optimizer.zero_grad()
        outputs = trainer.model(audio_in)
        loss = trainer.criterion(audio_in, outputs, **kwargs)
        loss.backward()
        trainer.optimizer.step()
        running_loss += loss.item()
    return running_loss


def simple_vqvae_loop(trainer: nn.Module, **kwargs) -> float:
    running_loss = 0.0
    for idx, data in enumerate(trainer.trainloader):
        audio_in, sr = data[0].to(trainer.device), data[1].to(trainer.device)
        trainer.optimizer.zero_grad()
        outputs, embeds, embed_loss, embed_fit = trainer.model(audio_in)
        loss = trainer.criterion(audio_in, outputs, **kwargs) + embed_loss
        loss.backward()
        trainer.optimizer.step()
        running_loss += loss.item()

        trainer.log(loss, "Batch loss", "scalar", step=idx)
        trainer.log(audio_in[0, :, :], "Audio input example", "audio")
        trainer.log(outputs[0, :, :], "Audio output example", "audio")
        trainer.log(embeds.view(-1, embeds.shape[1]*embeds.shape[2]), "Codebook Embedding", "embedding", step=idx)
    return running_loss
