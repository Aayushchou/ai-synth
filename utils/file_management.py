"""Utility module for handling file management and copy/pasting of the dataset into appropriate folders"""
import os
import shutil
import random


def sample_file_and_move(parent_dir: str, out_dir: str, n_files: int = 1) -> None:
    """Takes n number of files from all subdirectories within a given parent folder and copies to output director
    :param
        parent_dir: Directory containing all the subdirectories
        out_dir: Output directory where the files should be stored
        n_files: Number of files to take from each sub directory
    """

    print(f"Source directory: {parent_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Number of files to sample per directory: {n_files}")

    dirs = [x[1] for x in os.walk(parent_dir)]
    print(f"number of directories present: {len(dirs[0])}")

    for idx, dir in enumerate(dirs[0]):
        print(f"Directory {idx}: {dir}")
        files = os.listdir(os.path.join(parent_dir, dir))
        random_files = random.sample(files, n_files)
        in_files = [os.path.join(parent_dir, dir, f) for f in random_files]
        out_files = [os.path.join(out_dir, f"{dir.replace('-', '_')}__{f}") for f in random_files]
        _ = [shutil.copy(in_, out_) for in_, out_ in zip(in_files, out_files)]
    print(f"copy complete")
    
    
if __name__ == '__main__':
    sample_file_and_move(parent_dir=r"../resources/dx7_raw", out_dir="../resources/dx7")

