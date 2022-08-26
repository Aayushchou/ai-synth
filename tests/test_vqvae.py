import pytest
import torch.nn as nn
from torch import rand
from torch.optim import Adam

from aisynth.globals import TEST_INPUTS, VQVAE_CONFIG
from aisynth.model.autoencoder import EncoderBlock1D, DecoderBlock1D
from aisynth.model.vqvae import VQVAE
from aisynth.data.dataset import DX7SynthDataset
from aisynth.train.trainer import Trainer
from aisynth.train.losses import l1_loss
from aisynth.train.train_loop import simple_vqvae_loop


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
                            reverse_dilation=inputs["reverse_dilation"])
        model = nn.Sequential(eb, db)
        out = model.forward(inputs["test_tensor"])

        assert out.shape == inputs["test_tensor"].shape


class TestFullVQVAE:
    @pytest.mark.parametrize("inputs", [VQVAE_CONFIG])
    def test_init(self, inputs):
        """Tests the decoder function initialisation ensuring right architecture"""
        vqvae = VQVAE(autoencoder_args=inputs["autoencoder_args"],
                      quantizer_args=inputs["quantizer_args"])

    @pytest.mark.parametrize("inputs", [VQVAE_CONFIG])
    def test_forward(self, inputs):
        """Tests the forward function to ensure correct output shapes"""
        vqvae = VQVAE(autoencoder_args=inputs["autoencoder_args"],
                      quantizer_args=inputs["quantizer_args"])
        output, embeds, embed_loss, fit = vqvae(rand(8, 1, 2048))
        assert output.shape == (8, 1, 2048)

    @pytest.mark.parametrize("inputs", [VQVAE_CONFIG])
    def test_fit(self, inputs):
        vqvae = VQVAE(autoencoder_args=inputs["autoencoder_args"],
                      quantizer_args=inputs["quantizer_args"])
        sample_dataset = DX7SynthDataset(audio_dir=inputs["training_args"]["train_path"], duration=1.0)
        optimizer = Adam(vqvae.parameters())
        trainer = Trainer(model=vqvae,
                          train_dataset=sample_dataset,
                          device=inputs["training_args"]["device"],
                          optimizer=optimizer,
                          criterion=l1_loss,
                          n_epochs=inputs["training_args"]["n_epochs"],
                          train_loop=simple_vqvae_loop)
        trainer.log(trainer.model, "VQVAE Model", "model")
        trainer.fit()
