"""dataset.py:
    Definition of the DX7 synth dataset is provided here.
    The dataset consists of 4 second samples of different synth sounds.
    Joseph Turian. (2021). Timbre Audio Dataset (DX7-clone synthesizer) (1.0.0) [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.4677102"""
import os
import torchaudio
import pandas as pd
import tempfile

from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple
from functools import cached_property


class DX7SynthDataset(Dataset):

    def __init__(self, audio_dir, supported_formats=["ogg"]):
        """Initialise the dataset by providing the root path to the audio files."""
        super().__init__()
        self.audio_dir = audio_dir
        self.supported_formats = supported_formats

    @cached_property
    def metadata(self):
        return self._generate_metadata()

    def __len__(self) -> int:
        """Return the length of the dataset, which is the number of samples we have"""
        valid_samples = [file for file in os.listdir(self.audio_dir) if any(ext in file for ext in self.supported_formats)]
        return len(valid_samples)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        """Returns a sample from the dataset, audio sample and sample rate would be sensible
        :param
            index: The index of the sample to be returned
        """
        audio_sample_path = self.metadata.iloc[index, 2]
        signal, sr = torchaudio.load(audio_sample_path)

        return signal, signal

    def _generate_metadata(self) -> pd.DataFrame:
        """Utility function to create metadata for the audio file directory.
        Should be run each time the dataset is updated. Used for indexing mainly.
        :param
            audio_dir: Directory with the audio files
            metadata_filename: File name of the output metadata file
            supported_formats: Supported audio formats"""
        metadata_list = []
        for idx, file in enumerate(sorted(os.listdir(self.audio_dir))):
            if any(ext in file for ext in self.supported_formats):
                record_info = {}
                file_path = os.path.join(self.audio_dir, file)
                record_info["index"] = idx
                record_info["file_name"] = file
                record_info["full_path"] = os.path.abspath(file_path)
                record_info["ext"] = os.path.splitext(file)[-1]
                audio_metadata = torchaudio.info(file_path)
                record_info["sample_rate"] = audio_metadata.sample_rate
                record_info["num_frames"] = audio_metadata.num_frames
                record_info["num_channels"] = audio_metadata.num_channels
                record_info["bits_per_sample"] = audio_metadata.bits_per_sample
                record_info["encoding"] = audio_metadata.encoding
                metadata_list.append(record_info)

        md = pd.DataFrame(metadata_list)

        return md
