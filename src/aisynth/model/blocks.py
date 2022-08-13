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
    """Block for encoding an input audio signal into a latent embedding using resnets.

        params:
            input_emb_width: width of input audio
            output_emb_width: dimensions of the embedding layer
            down_t: Number of encoder blocks
            stride_t: stride of the conv1d layer before the resnet block
            width: width of the resnet blocks
            depth: number of resnet blocks
            m_conv: scaling factor to modify convolutional channels
            dilation_growth_rate: factor for growing the convolution dilation rate,
            dilation_cycle: integer to modulate the dilation factor around,
            zero_out: whether to zero out encoder outputs,
            res_scale: whether to scale residual outputs,
            dropout: dropout value for regularisation

        attrs:
            encoder_block: Model consisting of encoder blocks
    """
    def __init__(self,
                 input_emb_width: int,
                 output_emb_width: int,
                 down_t: int,
                 stride_t: int,
                 width: int,
                 depth: int,
                 m_conv: float,
                 dilation_growth_rate: int = 1,
                 dilation_cycle: bool = False,
                 zero_out: bool = False,
                 res_scale: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        filter_t, pad_t = stride_t * 2, stride_t // 2
        assert down_t > 0, f"Please ensure that the value of down_t is greater than 0"
        blocks = [nn.Sequential(nn.Conv1d(input_emb_width if i == 0 else width,
                                          width,
                                          (filter_t,),
                                          (stride_t,),
                                          pad_t),
                                ResidualBlock1D(n_in=width,
                                                n_depth=depth,
                                                m_conv=m_conv,
                                                dilation_growth_rate=dilation_growth_rate,
                                                dilation_cycle=dilation_cycle,
                                                zero_out=zero_out,
                                                scale=res_scale,
                                                dropout_val=dropout))
                  for i in range(down_t)]
        final_block = nn.Conv1d(width,
                                output_emb_width,
                                (3,),
                                (1,),
                                (1,))
        blocks.append(final_block)
        self.encoder_block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.encoder_block(x)


class DecoderBlock1D(nn.Module):
    """Block for decoding latent embedding back to an input audio signal using resnets and convolution transpose.

        params:
            input_emb_width: width of input audio
            output_emb_width: dimensions of the embedding layer
            down_t: Number of encoder blocks
            stride_t: stride of the conv1d layer before the resnet block
            width: width of the resnet blocks
            depth: number of resnet blocks
            m_conv: scaling factor to modify convolutional channels
            dilation_growth_rate: factor for growing the convolution dilation rate,
            dilation_cycle: integer to modulate the dilation factor around,
            zero_out: whether to zero out encoder outputs,
            res_scale: whether to scale residual outputs,
            dropout: dropout value for regularisation
            reverse_dilation: whether to reverse the order of dilation growth

        attrs:
            decoder_block: Model consisting of encoder blocks
    """
    def __init__(self,
                 input_emb_width: int,
                 output_emb_width: int,
                 down_t: int,
                 stride_t: int,
                 width: int,
                 depth: int,
                 m_conv: float,
                 dilation_growth_rate: int = 1,
                 dilation_cycle: bool = False,
                 zero_out: bool = False,
                 res_scale: bool = False,
                 dropout: float = 0.0,
                 reverse_dilation: bool = False):
        super().__init__()
        filter_t, pad_t = stride_t * 2, stride_t // 2
        assert down_t > 0, f"Please ensure that the value of down_t is greater than 0"
        blocks = [nn.Sequential(ResidualBlock1D(n_in=width,
                                                n_depth=depth,
                                                m_conv=m_conv,
                                                dilation_growth_rate=dilation_growth_rate,
                                                dilation_cycle=dilation_cycle,
                                                zero_out=zero_out,
                                                scale=res_scale,
                                                dropout_val=dropout,
                                                reverse_dilation=reverse_dilation),
                                nn.ConvTranspose1d(width,
                                                   input_emb_width if i == (down_t - 1) else width,
                                                   (filter_t,),
                                                   (stride_t,),
                                                   (pad_t,)),
                                )
                  for i in range(down_t)]
        first_block = nn.Conv1d(output_emb_width,
                                width,
                                (3,),
                                (1,),
                                (1,))
        blocks.insert(0, first_block)
        self.decoder_block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.decoder_block(x)
