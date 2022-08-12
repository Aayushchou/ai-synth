import torch.nn as nn
import math


class ConvBlock1D(nn.Module):
    """Convolutional block with a residual architecture in 1D. Used in the encoder and decoders.
    :param
        n_in -- number of input channels
        n_hidden -- number of hidden channels
        stride -- stride for the first convolutional layer
        padding -- padding for the first convolutional layer
        dilation -- dilation for the first convolutional layer
        scale -- scaling the model outputs to provide control over input/latent representation
        zero_out -- whether to output zeros
        dropout_flag -- including dropout in the block for regularisation
        dropout_val -- how much drop out to add
        batchnorm_flag -- whether to add batch normalisation
    : returns
        nn.Module object"""
    def __init__(self,
                 n_in,
                 n_hidden,
                 stride=1,
                 dilation=1,
                 scale=1.0,
                 zero_out=False,
                 dropout_flag=False,
                 dropout_val=0.3,
                 batchnorm_flag=False):
        """Setting up convolutional layers and normalisation based on parameters. """
        super().__init__()
        self.scale = scale
        self.dropout_flag = dropout_flag
        self.batchnorm_flag = batchnorm_flag

        padding = dilation
        self.layers = [
            nn.ReLU(),
            nn.Conv1d(n_in, n_hidden, 3, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(n_hidden, n_in, 1, stride=1, padding=0)
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

    def forward(self, x):
        """ Residual forward pass, input vector plus scaled hidden vector"""
        return x + self.scale * self.block(x)


class ResidualBlock1D(nn.Module):

    def __init__(self,
                 n_in,
                 n_depth,
                 dilation_growth_rate=1,
                 dilation_cycle=None,
                 zero_out=False,
                 dropout_val: int = 0,
                 m_conv=1.0,
                 scale=False,
                 reverse_dilation=False):
        super().__init__()
        self.dilation_cycle = dilation_cycle
        self.blocks = [ConvBlock1D(n_in=n_in,
                                   n_hidden=int(m_conv * n_in),
                                   dilation=dilation_growth_rate ** self._get_depth(depth),
                                   zero_out=zero_out,
                                   dropout_flag=False if dropout_val == 0 else True,
                                   dropout_val=dropout_val,
                                   scale=1.0 if not scale else 1.0 / math.sqrt(n_depth))
                       for depth in n_depth]

        if reverse_dilation:
            self.blocks = self.blocks[::-1]

        self.resnet = nn.Sequential(*self.blocks)

    def _get_depth(self, depth):
        return depth if self.dilation_cycle is None else depth % self.dilation_cycle

    def forward(self, x):
        # WATCH OUT FOR THIS addition, its what should be happening according to the diagram but isn't
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

