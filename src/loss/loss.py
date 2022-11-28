import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, pitch_pred, energy_pred,
                mel_target, duration_predictor_target, pitch_target, energy_target, offset=1.):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration_predictor_target.float())
        pitch_loss = self.mse_loss(pitch_pred, pitch_target)
        energy_loss = self.mse_loss(energy_pred, energy_target)

        return mel_loss, duration_predictor_loss, pitch_loss, energy_loss
