import torch.nn as nn
import torch


class ResidualBlock1D(nn.Module):

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
        super().__init__()
        self.scale = scale
        self.dropout_flag = dropout_flag
        self.batchnorm_flag = batchnorm_flag

        padding = dilation
        layers = [
            nn.ReLU(),
            nn.Conv1d(n_in, n_hidden, 3, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(n_hidden, n_in, 1, stride=1, padding=0)
        ]
        if batchnorm_flag:
            layers.insert(0, nn.BatchNorm1d(n_in))

        if dropout_flag:
            layers.insert(-1, nn.Dropout(dropout_val))

        self.block = nn.Sequential(*layers)

        if zero_out:
            out = self.block[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)

    def forward(self, x):
        return x + self.scale * self.block(x)


class EncoderBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class DecoderBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

