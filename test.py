import numpy as np
import torch
from tqdm import tqdm

import argparse
from pathlib import Path
import json

from src import hparams as hp
from src.text import text_to_sequence
from src.model.FastSpeech2 import FastSpeech
from src.audio.tools import inv_mel_spec
from src import waveglow
from src.utils import get_WaveGlow

def process_texts(texts):
    data_list = list(text_to_sequence(text, hp.text_cleaners) for text in texts)

    return data_list

def mel_synthesis(model, text, device, alpha=1., beta=1., gamma=1.):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, beta=beta, gamma=gamma)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)

def synthesis(model, texts, device, params, args, WaveGlow):
    texts = process_texts(texts)
    for i, text in enumerate(tqdm(texts)):
        for alpha, beta, gamma in zip(params["alpha"], params["beta"], params["gamma"]):
            mel, mel_cuda = mel_synthesis(model, text, device, alpha, beta, gamma)
            directory = Path(args.results)
            directory.mkdir(exist_ok=True)
            # inv_mel_spec(mel, directory / f"{i}_a{alpha}_b{beta}_g{gamma}.wav")
            waveglow.inference.inference(mel_cuda, WaveGlow, directory / f"{i}_a{alpha}_b{beta}_g{gamma}_waveglow.wav")
                    

def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-m",
        "--model",
        default=None,
        type=str,
        help="model checkpoint file path (default: None)",
    )
    args.add_argument(
        "-t",
        "--text",
        default=None,
        type=str,
        help="text file path to do predict on",
    )
    args.add_argument(
        "-p",
        "--params",
        default="synthesis_params.json",
        type=str,
        help="synthesis parameters json file path (default: synthesis_params.json)",
    )
    args.add_argument(
        "-r",
        "--results",
        default="results",
        type=str,
        help="the directory to write the results to"
    )
    args = args.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_t = torch.tensor(1)
    dummy_stats = {"f0": {"min": dummy_t, "max": dummy_t}, "energy": {"min": dummy_t, "max": dummy_t}}
    model = FastSpeech(hp, dummy_stats)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    with open(args.params) as file:
        params = json.load(file)
    with open(args.text, "r") as file:
        texts = file.readlines()
    WaveGlow = get_WaveGlow()
    synthesis(model, texts, device, params, args, WaveGlow)  

if __name__ == "__main__":
    main()