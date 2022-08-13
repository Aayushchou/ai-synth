import pytest
from aisynth.globals import TEST_INPUTS
from aisynth.model.blocks import ConvBlock1D, ResidualBlock1D, EncoderBlock1D, DecoderBlock1D


class TestConvBlock:
    """Test suite for checking validity of the convolutional block object."""

    @pytest.mark.parametrize("inputs", TEST_INPUTS["CONVBLOCK_TEST"])
    def test_init(self, inputs):
        """Tests initialisation and whether the metadata is generated appropriately"""
        rb = ConvBlock1D(n_in=inputs["n_in"],
                         n_hidden=inputs["n_hidden"],
                         batchnorm_flag=inputs["batchnorm_flag"],
                         dropout_flag=inputs["dropout_flag"])
        assert len(rb.layers) in [4, 6]

    @pytest.mark.parametrize("inputs", TEST_INPUTS["CONVBLOCK_TEST"])
    def test_forward(self, inputs):
        """Tests length of the dataset and whether it matches what we expect"""
        rb = ConvBlock1D(n_in=inputs["n_in"],
                         n_hidden=inputs["n_hidden"],
                         batchnorm_flag=inputs["batchnorm_flag"],
                         dropout_flag=inputs["dropout_flag"])

        out = rb.forward(inputs["test_tensor"])
        assert out.shape == inputs["test_tensor"].shape


class TestResidualBlock:
    """Test suite for checking validity of the residual block object."""

    @pytest.mark.parametrize("inputs", TEST_INPUTS["RESBLOCK_TEST"])
    def test_init(self, inputs):
        """Tests initialisation and whether the metadata is generated appropriately"""
        rb = ResidualBlock1D(n_in=inputs["n_in"],
                             n_depth=inputs["n_depth"],
                             dilation_cycle=inputs["dilation_cycle"],
                             dilation_growth_rate=inputs["dilation_growth_rate"],
                             reverse_dilation=inputs["reverse_dilation"])
        assert len(rb.blocks) == inputs["n_depth"]

    @pytest.mark.parametrize("inputs", TEST_INPUTS["RESBLOCK_TEST"])
    def test_forward(self, inputs):
        """Tests length of the dataset and whether it matches what we expect"""
        rb = ResidualBlock1D(n_in=inputs["n_in"],
                             n_depth=inputs["n_depth"],
                             dilation_cycle=inputs["dilation_cycle"],
                             dilation_growth_rate=inputs["dilation_growth_rate"],
                             reverse_dilation=inputs["reverse_dilation"],
                             m_conv=inputs["m_conv"])

        out = rb.forward(inputs["test_tensor"])
        assert out.shape == inputs["test_tensor"].shape


class TestEncoderBlock:
    @pytest.mark.parametrize("inputs", TEST_INPUTS["RESBLOCK_TEST"])
    def test_init(self, inputs):
        """Tests initialisation and whether the metadata is generated appropriately"""
        rb = ResidualBlock1D(n_in=inputs["n_in"],
                             n_depth=inputs["n_depth"],
                             dilation_cycle=inputs["dilation_cycle"],
                             dilation_growth_rate=inputs["dilation_growth_rate"],
                             reverse_dilation=inputs["reverse_dilation"])
        assert len(rb.blocks) == inputs["n_depth"]

    @pytest.mark.parametrize("inputs", TEST_INPUTS["RESBLOCK_TEST"])
    def test_forward(self, inputs):
        """Tests length of the dataset and whether it matches what we expect"""
        rb = ResidualBlock1D(n_in=inputs["n_in"],
                             n_depth=inputs["n_depth"],
                             dilation_cycle=inputs["dilation_cycle"],
                             dilation_growth_rate=inputs["dilation_growth_rate"],
                             reverse_dilation=inputs["reverse_dilation"],
                             m_conv=inputs["m_conv"])

        out = rb.forward(inputs["test_tensor"])
        assert out.shape == inputs["test_tensor"].shape


class TestDecoderBlock:
    @pytest.mark.parametrize("inputs", TEST_INPUTS["RESBLOCK_TEST"])
    def test_init(self, inputs):
        """Tests initialisation and whether the metadata is generated appropriately"""
        rb = ResidualBlock1D(n_in=inputs["n_in"],
                             n_depth=inputs["n_depth"],
                             dilation_cycle=inputs["dilation_cycle"],
                             dilation_growth_rate=inputs["dilation_growth_rate"],
                             reverse_dilation=inputs["reverse_dilation"])
        assert len(rb.blocks) == inputs["n_depth"]

    @pytest.mark.parametrize("inputs", TEST_INPUTS["RESBLOCK_TEST"])
    def test_forward(self, inputs):
        """Tests length of the dataset and whether it matches what we expect"""
        rb = ResidualBlock1D(n_in=inputs["n_in"],
                             n_depth=inputs["n_depth"],
                             dilation_cycle=inputs["dilation_cycle"],
                             dilation_growth_rate=inputs["dilation_growth_rate"],
                             reverse_dilation=inputs["reverse_dilation"],
                             m_conv=inputs["m_conv"])

        out = rb.forward(inputs["test_tensor"])
        assert out.shape == inputs["test_tensor"].shape
