import torch.nn as nn

from aisynth.model.autoencoder import EncoderBlock1D, DecoderBlock1D
from aisynth.model.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    """Container class for the overall VQ-VAE model, including the encoder, decoder and quantizer.
    params:
        encoder_args: dict"""

    def __init__(self, autoencoder_args: dict, quantizer_args: dict):

        super().__init__()
        self.encoder = EncoderBlock1D(**autoencoder_args)
        self.quantizer = VectorQuantizer(**quantizer_args)
        self.decoder = DecoderBlock1D(**autoencoder_args)

    def forward(self, x):
        x = self.encoder(x)
        q, q_loss, c_fit = self.quantizer(x)
        x_out = self.decoder(x)

        return x_out, q, q_loss, c_fit
