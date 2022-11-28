import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gdown

from src import audio
import src.hparams
from src import glow


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


def get_WaveGlow(): #1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
    waveglow_path = Path("waveglow") / "pretrained_model"
    if not waveglow_path.exists():
        gdown.download(id="1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx")
        #mkdir -p waveglow/pretrained_model/
        #mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt
        waveglow_path.mkdir(parents=True)
        waveglow_path = Path("waveglow_256channels_ljs_v2.pt").rename(waveglow_path / "waveglow_256channels.pt")
    waveglow_path = waveglow_path / "waveglow_256channels.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wave_glow = torch.load(waveglow_path, map_location=device)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return wave_glow


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        out_list = list()
        max_len = mel_max_length
        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    else:
        out_list = list()
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    
    
def log_predictions(
    logger,
    preds,
    targets,
    examples_to_log=4,
    *args,
    **kwargs
    ):
    rows = {}
    index = 0
    for pred, target in zip(preds, targets):
        if index >= examples_to_log:
            break
        
        pred = audio.tools.inv_mel_spec(pred.cpu().transpose(0, 1))
        target = audio.tools.inv_mel_spec(target.cpu().transpose(0, 1))
        pred = logger.wandb.Audio(pred, sample_rate=audio.hparams_audio.sampling_rate)
        target = logger.wandb.Audio(target, sample_rate=audio.hparams_audio.sampling_rate)
    
        rows[index] = {
            "ground_truth": target,
            "prediction": pred
        }
        index += 1
    logger.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))
