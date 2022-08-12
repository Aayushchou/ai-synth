import torch.nn as nn
import torch
import math


class ConvBlock1D(nn.Module):
    """
    Convolutional block with a residual architecture in 1D. Used in the encoder and decoders.

            params:
                n_in: number of input channels
                n_hidden: number of hidden channels
                stride: stride for the first convolutional layer
                padding: padding for the first convolutional layer
                dilation: dilation for the first convolutional layer
                scale: scaling the model outputs to provide control over input/latent representation
                zero_out: whether to output zeros
                dropout_flag: including dropout in the block for regularisation
                dropout_val: how much drop out to add
                batchnorm_flag: whether to add batch normalisation

            attr:
                scale: Scaling factor of the residual block outputs
                dropout_flag: Whether to add dropout
                batchnorm_flag: Whether to add batch normalisation
                layers: List containing layers in model
                block: Module object with user defined layers

    """
    def __init__(self,
                 n_in: int,
                 n_hidden: int,
                 stride: int = 1,
                 dilation: int = 1,
                 scale: float = 1.0,
                 zero_out: bool = False,
                 dropout_flag: bool = False,
                 dropout_val: float = 0.3,
                 batchnorm_flag: bool = False):
        """Setting up convolutional layers and normalisation based on parameters. """
        super().__init__()
        self.scale = scale
        self.dropout_flag = dropout_flag
        self.batchnorm_flag = batchnorm_flag

        padding = dilation
        self.layers = [
            nn.ReLU(),
            nn.Conv1d(n_in, n_hidden, (3,), stride=(stride,), padding=padding, dilation=(dilation,)),
            nn.ReLU(),
            nn.Conv1d(n_hidden, n_in, (1,), stride=(1,), padding=0)
        ]
        if batchnorm_flag:
            self.layers.insert(0, nn.BatchNorm1d(n_in))

        if dropout_flag:
            self.layers.insert(-1, nn.Dropout(dropout_val))

        self.block = nn.Sequential(*self.layers)

        if zero_out:
            out = self.block[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Residual forward pass, input vector plus scaled hidden vector"""
        return x + self.scale * self.block(x)


class ResidualBlock1D(nn.Module):
    """
    Residual block with dilated convolutions. Used in the encoder and decoders.

            params:
                n_in: number of input channels
                n_depth: depth of the resnet
                dilation_growth_rate: factor by which the dilation increases or decreases
                dilation_cycle: if diilation should modulate around a value
                zero_out: whether to output zeros
                dropout_val: value of dropout for regularization
                m_conv: scaling factor for hidden convolution dimension
                scale: scaling factor on how to scale residual outputs
                reverse_dilation: whether to reverse dilation pattern, used for decoder
                batchnorm_flag: whether to add batch normalisation

            attr:
                blocks: list of convolutional blocks with specified parameters
                resnet: torch sequential module for the resnet

    """
    def __init__(self,
                 n_in: int,
                 n_depth: int,
                 dilation_growth_rate: int = 1,
                 dilation_cycle: int = None,
                 zero_out: bool = False,
                 dropout_val: float = 0.0,
                 m_conv: float = 1.0,
                 scale: bool = False,
                 reverse_dilation: bool = False):
        """Defines list of residual convolutional blocks for the network. """
        super().__init__()
        self.dilation_cycle = dilation_cycle
        self.blocks = [ConvBlock1D(n_in=n_in,
                                   n_hidden=int(m_conv * n_in),
                                   dilation=dilation_growth_rate ** self._get_depth(depth),
                                   zero_out=zero_out,
                                   dropout_flag=False if dropout_val == 0.0 else True,
                                   dropout_val=dropout_val,
                                   scale=1.0 if not scale else 1.0 / math.sqrt(n_depth))
                       for depth in range(n_depth)]

        if reverse_dilation:
            self.blocks = self.blocks[::-1]

        self.resnet = nn.Sequential(*self.blocks)

    def _get_depth(self, depth: int) -> int:
        """returns depth based on the dilation cycle for calculating dilation"""
        return depth if self.dilation_cycle is None else depth % self.dilation_cycle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass through the model"""
        # WATCH OUT FOR THIS ADDITION
        return x + self.resnet(x)


class EncoderBlock1D(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class DecoderBlock1D(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

