import shutil
import os
from pathlib import Path

import torch
import numpy as np
from speechbrain.utils.data_utils import download_file

from src.dataset import ljspeech
import src.hparams as hp


def preprocess_ljspeech(filename):
    in_dir = filename
    out_dir = hp.mel_ground_truth
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    metadata = ljspeech.build_from_path(in_dir, out_dir)
    write_metadata(metadata, out_dir)

    shutil.move(os.path.join(hp.mel_ground_truth, "train.txt"),
                os.path.join("data", "train.txt"))


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write(m + '\n')
            
def download_ljspeech():
    ljspeech_dir = Path(hp.data_path) / "LJSpeech-1.1"
    archive = Path(hp.data_path) / "LJSpeech-1.1.tar.bz2"
    if not ljspeech_dir.exists():
        download_file("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", dest=archive,
                      unpack = True, dest_unpack=ljspeech_dir.parent)
    return ljspeech_dir
        


def main():
    ljspeech_dir = download_ljspeech()
    preprocess_ljspeech(ljspeech_dir)


if __name__ == "__main__":
    main()
