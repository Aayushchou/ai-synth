import pytest
import torch.nn as nn
from torch.optim import Adam

from aisynth.train.train import Trainer
from aisynth.data.dataset import DX7SynthDataset
from aisynth.utils.losses import l1_loss
from aisynth.model.blocks import EncoderBlock1D, DecoderBlock1D
from aisynth.train.one_epoch import simple_train_loop
from aisynth.globals import TEST_INPUTS


class TestTraining:

    @pytest.mark.parametrize("inputs", TEST_INPUTS["TRAIN_TEST"])
    def test_init(self, inputs):
        """Tests the initialisation of the training module."""
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
        sample_dataset = DX7SynthDataset(audio_dir=inputs["audio_dir"], duration=1.0)
        optimizer = Adam(model.parameters())
        trainer = Trainer(model=model,
                          train_dataset=sample_dataset,
                          device="cpu",
                          optimizer=optimizer,
                          n_epochs=inputs["n_epochs"],
                          criterion=l1_loss,
                          epoch_func=simple_train_loop)
        print(trainer.__dict__)

    @pytest.mark.parametrize("inputs", TEST_INPUTS["TRAIN_TEST"])
    def test_fit(self, inputs):
        """Tests the entire training process to ensure that its functional."""
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
        sample_dataset = DX7SynthDataset(audio_dir=inputs["audio_dir"], duration=1.0)
        optimizer = Adam(model.parameters())
        trainer = Trainer(model=model,
                          train_dataset=sample_dataset,
                          device="cpu",
                          optimizer=optimizer,
                          n_epochs=inputs["n_epochs"],
                          criterion=l1_loss,
                          epoch_func=simple_train_loop)
        trainer.fit()
        test_out = trainer.forward(sample_dataset[1][0])
        loss = l1_loss(test_out, sample_dataset[1][0])
        print(loss.item())
