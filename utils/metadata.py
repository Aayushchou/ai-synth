"""Script for creating a metadata file for the audio tracks, for the purposes of indexing and reproducibility """
import os
import pandas as pd


def create_metadata(audio_dir: str, metadata_filename: str, supported_formats=["ogg"]) -> None:
    """Utility function to create metadata for the audio file directory.
    Should be run each time the dataset is updated. Used for indexing mainly.
    :param
        audio_dir: Directory with the audio files
        metadata_filename: File name of the output metadata file
        supported_formats: Supported audio formats"""
    metadata_list = []
    print(f"Audio directory: {audio_dir}")
    print(f"Metadata directory: {metadata_filename}"
          )
    print(f"index--file_name--full_path--ext")
    for idx, file in enumerate(sorted(os.listdir(audio_dir))):
        if any(ext in file for ext in supported_formats):
            record_info = {}
            file_path = os.path.join(audio_dir, file)
            record_info["index"] = idx
            record_info["file_name"] = file
            record_info["full_path"] = os.path.abspath(file_path)
            record_info["ext"] = os.path.splitext(file)[-1]
            print(idx, file)
            metadata_list.append(record_info)

    md = pd.DataFrame(metadata_list)
    md.to_csv(metadata_filename, index=False)


if __name__ == '__main__':
    create_metadata(audio_dir="resources/dx7", metadata_filename="resources/dx7.csv")




