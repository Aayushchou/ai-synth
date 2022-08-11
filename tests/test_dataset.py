import pytest
from aisynth.globals import AUDIO_DIR_TEST
from aisynth.dataset import DX7SynthDataset


class TestSynthDX7Dataset:
    """Test suite for checking validity of the DX7 dataset object.
    If the dataset is updated, these tests need to be updated. """

    @pytest.mark.parametrize("inputs", AUDIO_DIR_TEST)
    def test_init(self, inputs):
        """Tests initialisation and whether the metadata is generated appropriately"""
        dx = DX7SynthDataset(audio_dir=inputs["audio_dir"])
        assert dx.metadata.shape == (3089, 9)

    @pytest.mark.parametrize("inputs", AUDIO_DIR_TEST)
    def test_len(self, inputs):
        """Tests length of the dataset and whether it matches what we expect"""
        dx = DX7SynthDataset(audio_dir=inputs["audio_dir"])
        assert len(dx) == 3089

    @pytest.mark.parametrize("inputs", AUDIO_DIR_TEST)
    def test_get_items(self, inputs):
        """Tests whether we are able to extract items appropriately. """
        dx = DX7SynthDataset(audio_dir=inputs["audio_dir"])
        item_7 = dx[7]
        assert item_7[0].shape == item_7[1].shape
        assert item_7[0].shape == (1, 176400)