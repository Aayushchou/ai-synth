import pytest
import torch.nn as nn
from aisynth.globals import TEST_INPUTS
from aisynth.model.blocks import EncoderBlock1D, DecoderBlock1D


class TestEncoderDecoder:

    @pytest.mark.parametrize("inputs", TEST_INPUTS["ENCDEC_TEST"])
    def test_init(self, inputs):
        """Tests the decoder function initialisation ensuring right architecture"""
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
        model = nn.Sequential(eb, db)

    @pytest.mark.parametrize("inputs", TEST_INPUTS["ENCDEC_TEST"])
    def test_forward(self, inputs):
        """Tests the forward function to ensure correct output shapes"""
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
                            batchnorm_flag=inputs["batchnorm_flag"],
                            reverse_dilation=inputs["reverse_dilation"])
        model = nn.Sequential(eb, db)
        out = model.forward(inputs["test_tensor"])

        assert out.shape == inputs["test_tensor"].shape
