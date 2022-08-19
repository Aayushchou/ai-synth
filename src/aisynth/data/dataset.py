import os
import torchaudio
import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Tuple, Callable
from functools import cached_property

from aisynth.data.preprocess import resample


class DX7SynthDataset(Dataset):
    """
    Definition of the DX7 synth dataset is provided here.
    The dataset consists of 4 second samples of different synth sounds.
    Joseph Turian. (2021). Timbre Audio Dataset (DX7-clone synthesizer) (1.0.0) [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.4677102
    """

    def __init__(
        self,
        audio_dir: str,
        supported_formats: Tuple[str] = ("ogg",),
        duration: float = None,
        target_sr: int = 22500,
        transformation: Callable[[Tensor, int], Tensor] = None,
    ):
        """Initialise the dataset by providing the root path to the audio files."""
        super().__init__()
        self.audio_dir = audio_dir
        self.supported_formats = supported_formats
        self.transformation = transformation
        self.duration = duration
        self.target_sr = target_sr

    @cached_property
    def metadata(self):
        return self._generate_metadata()

    def __len__(self) -> int:
        """Return the length of the dataset, which is the number of samples we have"""
        valid_samples = [
            file
            for file in os.listdir(self.audio_dir)
            if any(ext in file for ext in self.supported_formats)
        ]
        return len(valid_samples)

    def __getitem__(self, index) -> Tuple[Tensor, int]:
        """Returns a sample from the dataset, audio sample and sample rate would be sensible
        :param
            index: The index of the sample to be returned
        :returns
            A tuple of the same audio waveform, as we are creating an autoencoder
        """
        audio_sample_path = self.metadata.iloc[index, 2]
        signal, sr = torchaudio.load(audio_sample_path)
        signal, sr = self._transform_audio(signal, sr)
        return signal, sr

    def _transform_audio(self, signal: Tensor, sr: int) -> Tuple[Tensor, int]:
        """Function for pre-processing and transforming audio to the desired state.
        params:
            signal: The input audio
            sr: The input sample rate
        procedure:
            1. resamples audio to the target sample rate
            2. transforms audio based on given audio transformation.
            3. changes audio duration based on the duration input.
            4. right pads audio if necessary
        returns:
            processed audio and sample rate."""
        if sr != self.target_sr:
            signal = resample(signal, sr, self.target_sr)
            sr = self.target_sr
        if self.transformation:
            signal = self.transformation(signal, sr)
        if self.duration:
            cut_point = int(self.duration * sr)
            signal = signal[:, :cut_point]

        signal = self._fix_length(signal)
        return signal, sr

    def _fix_length(self, signal: Tensor) -> Tensor:
        """right pad audio to the desired length."""
        if signal.shape[1] < self.target_sr:
            missing_vals = int(self.target_sr * self.duration) - signal.shape[1]
            signal = F.pad(signal, (0, missing_vals))
        return signal

    def _generate_metadata(self) -> pd.DataFrame:
        """Utility function to create metadata for the audio file directory.
        Should be run each time the dataset is updated. Used for indexing mainly.
        :param
            audio_dir: Directory with the audio files
            metadata_filename: File name of the output metadata file
            supported_formats: Supported audio formats
        :returns
            A DataFrame containing file level and audio level metadata
        """
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
