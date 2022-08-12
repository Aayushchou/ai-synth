import torch.nn as nn


class ResidualBlock1D(nn.Module):
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

