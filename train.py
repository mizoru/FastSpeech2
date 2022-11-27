import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler  import OneCycleLR
from tqdm import tqdm


from src.dataset import get_dataset
from src.model.FastSpeech2 import FastSpeech
from src import hparams as hp
from src.loss.loss import FastSpeechLoss
from src.visualizer.wandb import WanDBWriter
from src.collate import collate_fn
from src.utils import log_predictions


dataset = get_dataset()

training_loader = DataLoader(
    dataset,
    batch_size=hp.batch_expand_size * hp.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=0
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FastSpeech(hp, dataset.stats)
criterion = FastSpeechLoss()
logger = WanDBWriter(hp)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=hp.learning_rate,
    betas=(0.9, 0.98),
    eps=1e-9)

current_step = 0

scheduler = OneCycleLR(optimizer, **{
    "steps_per_epoch": len(training_loader) * hp.batch_expand_size,
    "epochs": hp.epochs,
    "anneal_strategy": "cos",
    "max_lr": hp.learning_rate,
    "pct_start": 0.1
})

tqdm_bar = tqdm(total=hp.epochs * len(training_loader) * hp.batch_expand_size - current_step)

model.train()

new_steps = 0

for epoch in range(hp.epochs):
    for i, batchs in enumerate(training_loader):
        # real batch start here
        for j, db in enumerate(batchs):
            new_steps += 1
            tqdm_bar.update(1)
            if new_steps < -1:
                continue
            current_step += 1

            logger.set_step(current_step)
            # Get Data
            character = db["text"].long().to(hp.device)
            mel_target = db["mel_target"].float().to(hp.device)
            duration = db["duration"].int().to(hp.device)
            pitch = db["f0"].to(hp.device)
            energy = db["energy"].to(hp.device)
            mel_pos = db["mel_pos"].long().to(hp.device)
            src_pos = db["src_pos"].long().to(hp.device)
            max_mel_len = db["mel_max_len"]

            # Forward
            mel_output, duration_predictor_output, pitch_prediction, energy_prediction = model(character,
                                                          src_pos,
                                                          mel_pos=mel_pos,
                                                          mel_max_length=max_mel_len,
                                                          length_target=duration,
                                                          pitch_target=pitch,
                                                          energy_target=energy
                                                          )

            # Cal Loss
            mel_loss, duration_loss, pitch_loss, energy_loss = criterion(mel_output,
                                                    duration_predictor_output,
                                                    pitch_prediction,
                                                    energy_prediction,
                                                    mel_target,
                                                    duration,
                                                    pitch,
                                                    energy)
            total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

            # Logger
            t_l = total_loss.detach().cpu().numpy()
            m_l = mel_loss.detach().cpu().numpy()
            d_l = duration_loss.detach().cpu().numpy()
            p_l = pitch_loss.detach().cpu().numpy()
            e_l = energy_loss.detach().cpu().numpy()

            logger.add_scalar("duration_loss", p_l)
            logger.add_scalar("pitch_loss", d_l)
            logger.add_scalar("energy_loss", e_l)
            logger.add_scalar("mel_loss", m_l)
            logger.add_scalar("total_loss", t_l)

            # Backward
            total_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(
                model.parameters(), hp.grad_clip_thresh)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if current_step % hp.save_step == 0:
                os.makedirs(hp.checkpoint_path, exist_ok=True)
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth' % current_step))
                print("save model at step %d ..." % current_step)
