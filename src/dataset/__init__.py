import time
import os

import torch
from torch.utils.data import Dataset
import tqdm
import numpy as np

from src.utils import process_text
from src.text import text_to_sequence
from src.dataset.preprocess import main as preprocess
import src.hparams as hp

class BufferDataset(Dataset):
    def __init__(self, buffer, stats):
        self.buffer = buffer
        self.stats = stats
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]

def get_data_to_buffer():
    buffer = list()
    text = process_text(hp.data_path+"/train.txt")

    min_f0 = torch.inf
    max_f0 = -torch.inf

    min_energy = torch.inf
    max_energy = -torch.inf

    start = time.perf_counter()
    for i in tqdm(range(1, len(text)+1)):

        mel_gt_name = os.path.join(
            hp.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            hp.alignment_path, str(i)+".npy"))
        f0 = np.load(os.path.join(
            hp.f0s_path, str(i)+".npy")).astype(np.float32)
        energy = np.load(os.path.join(
            hp.energies_path, str(i)+".npy"))
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, hp.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        f0 = torch.from_numpy(f0)
        energy = torch.from_numpy(energy)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        f0_min = f0.min()
        min_f0 = f0_min if f0_min < min_f0 else min_f0
        f0_max = f0.max()
        max_f0 = f0_max if f0_max > max_f0 else max_f0

        energy_min = energy.min()
        min_energy = energy_min if energy_min < min_energy else min_energy
        energy_max = energy.max()
        max_energy = energy_max if energy_max > max_energy else max_energy

        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target, "f0":f0, "energy": energy})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    stats = {"f0": {"min": min_f0, "max": max_f0}, "energy": {"min": min_energy, "max": max_energy}}

    return buffer, stats
   
def get_dataset():
    preprocess()
    buffer, stats = get_data_to_buffer()
    return BufferDataset(buffer, stats)