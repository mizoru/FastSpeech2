from pathlib import Path
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
import numpy as np
import librosa
import pyworld as pw

from src import audio
from src.audio import hparams_audio
import src.hparams as hp



def build_from_path(in_dir, out_dir):
    index = 1
    # executor = ProcessPoolExecutor(max_workers=4)
    # futures = []
    texts = []

    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f.readlines():
            if index % 100 == 0:
                print("{:d} Done".format(index))
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            # futures.append(executor.submit(
            #     partial(_process_utterance, out_dir, index, wav_path, text)))
            texts.append(_process_utterance(out_dir, index, wav_path, text))

            index = index + 1

    # return [future.result() for future in tqdm(futures)]
    return texts

frame_period = hparams_audio.hop_length / hparams_audio.sampling_rate * 1000

def _process_utterance(out_dir, index, wav_path, text):
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram, energy = audio.tools.get_mel(wav_path)
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)
    energy = energy.numpy().astype(np.float32)
    # Write the spectrograms to disk:
    mel_filename = 'ljspeech-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.T, allow_pickle=False)
    
    wav, sr = librosa.load(wav_path, sr=None, dtype=np.float64)
    f0, t = pw.dio(wav, sr, frame_period=frame_period)
    assert f0.shape[0] == mel_spectrogram.shape[0]
    np.save(hp.f0s_path+f"/{index}", f0)
    np.save(hp.energies_path+f"/{index}", energy)

    return text
