import pytest
from aisynth.globals import TEST_INPUTS
from aisynth.model.autoencoder import ConvBlock1D, ResidualBlock1D, EncoderBlock1D, DecoderBlock1D


class TestConvBlock:
    """Test suite for checking validity of the convolutional block object."""

    @pytest.mark.parametrize("inputs", TEST_INPUTS["CONVBLOCK_TEST"])
    def test_init(self, inputs):
        """Tests initialisation and whether the correct amount of layers are being built"""
        rb = ConvBlock1D(n_in=inputs["n_in"],
                         n_hidden=inputs["n_hidden"],
                         dropout_flag=inputs["dropout_flag"])
        assert len(rb.layers) in [4, 5]

    @pytest.mark.parametrize("inputs", TEST_INPUTS["CONVBLOCK_TEST"])
    def test_forward(self, inputs):
        """Tests forward function and whether the output is of the shape we expect"""
        rb = ConvBlock1D(n_in=inputs["n_in"],
                         n_hidden=inputs["n_hidden"],
                         dropout_flag=inputs["dropout_flag"])

        out = rb.forward(inputs["test_tensor"])
        assert out.shape == inputs["test_tensor"].shape


class TestResidualBlock:
    """Test suite for checking validity of the residual block object."""

    @pytest.mark.parametrize("inputs", TEST_INPUTS["RESBLOCK_TEST"])
    def test_init(self, inputs):
        """Tests initialisation and whether the residual block is getting the right depth of blocks"""
        rb = ResidualBlock1D(n_in=inputs["n_in"],
                             n_depth=inputs["n_depth"],
                             dilation_cycle=inputs["dilation_cycle"],
                             dilation_growth_rate=inputs["dilation_growth_rate"],
                             reverse_dilation=inputs["reverse_dilation"])
        assert len(rb.blocks) == inputs["n_depth"]

    @pytest.mark.parametrize("inputs", TEST_INPUTS["RESBLOCK_TEST"])
    def test_forward(self, inputs):
        """Tests forward function and whether the output shape is what we expect"""
        rb = ResidualBlock1D(n_in=inputs["n_in"],
                             n_depth=inputs["n_depth"],
                             dilation_cycle=inputs["dilation_cycle"],
                             dilation_growth_rate=inputs["dilation_growth_rate"],
                             reverse_dilation=inputs["reverse_dilation"],
                             m_conv=inputs["m_conv"])

        out = rb.forward(inputs["test_tensor"])
        assert out.shape == inputs["test_tensor"].shape


class TestEncoderBlock:
    @pytest.mark.parametrize("inputs", TEST_INPUTS["ENCODERBLOCK_TEST"])
    def test_init(self, inputs):
        """Tests the encoder block initialisation and whether it is as we expect"""
        eb = EncoderBlock1D(input_emb_width=inputs["input_emb_width"],
                            output_emb_width=inputs["output_emb_width"],
                            down_t=inputs["down_t"],
                            stride_t=inputs["stride_t"],
                            width=inputs["width"],
                            depth=inputs["depth"],
                            m_conv=inputs["m_conv"],
                            dilation_cycle=inputs["dilation_cycle"],
                            dilation_growth_rate=inputs["dilation_growth_rate"],
                            zero_out=inputs["zero_out"],
                            res_scale=inputs["res_scale"],
                            dropout=inputs["dropout"])

        assert len(eb.encoder_block)-1 == inputs["down_t"]
        assert len(eb.encoder_block[0][1].resnet) == inputs["depth"]

    @pytest.mark.parametrize("inputs", TEST_INPUTS["ENCODERBLOCK_TEST"])
    def test_forward(self, inputs):
        """Tests the forward function to ensure that latent features have the right shapes"""
        eb = EncoderBlock1D(input_emb_width=inputs["input_emb_width"],
                            output_emb_width=inputs["output_emb_width"],
                            down_t=inputs["down_t"],
                            stride_t=inputs["stride_t"],
                            width=inputs["width"],
                            depth=inputs["depth"],
                            m_conv=inputs["m_conv"],
                            dilation_cycle=inputs["dilation_cycle"],
                            dilation_growth_rate=inputs["dilation_growth_rate"],
                            zero_out=inputs["zero_out"],
                            res_scale=inputs["res_scale"],
                            dropout=inputs["dropout"])

        out = eb.forward(inputs["test_tensor"])
        assert out.shape[0] == inputs["output_emb_width"]


class TestDecoderBlock:
    @pytest.mark.parametrize("inputs", TEST_INPUTS["DECODERBLOCK_TEST"])
    def test_init(self, inputs):
        """Tests the decoder function initialisation ensuring right architecture"""
        db = DecoderBlock1D(input_emb_width=inputs["input_emb_width"],
                            output_emb_width=inputs["output_emb_width"],
                            down_t=inputs["down_t"],
                            stride_t=inputs["stride_t"],
                            width=inputs["width"],
                            depth=inputs["depth"],
                            m_conv=inputs["m_conv"],
                            dilation_cycle=inputs["dilation_cycle"],
                            dilation_growth_rate=inputs["dilation_growth_rate"],
                            zero_out=inputs["zero_out"],
                            res_scale=inputs["res_scale"],
                            dropout=inputs["dropout"],
                            reverse_dilation=inputs["reverse_dilation"])

        assert len(db.decoder_block) - 1 == inputs["down_t"]

    @pytest.mark.parametrize("inputs", TEST_INPUTS["DECODERBLOCK_TEST"])
    def test_forward(self, inputs):
        """Tests the forward function to ensure correct output shapes"""
        db = DecoderBlock1D(input_emb_width=inputs["input_emb_width"],
                            output_emb_width=inputs["output_emb_width"],
                            down_t=inputs["down_t"],
                            stride_t=inputs["stride_t"],
                            width=inputs["width"],
                            depth=inputs["depth"],
                            m_conv=inputs["m_conv"],
                            dilation_cycle=inputs["dilation_cycle"],
                            dilation_growth_rate=inputs["dilation_growth_rate"],
                            zero_out=inputs["zero_out"],
                            res_scale=inputs["res_scale"],
                            dropout=inputs["dropout"],
                            reverse_dilation=inputs["reverse_dilation"])

        out = db.forward(inputs["test_tensor"])
        assert out.shape[0] == inputs["input_emb_width"]

