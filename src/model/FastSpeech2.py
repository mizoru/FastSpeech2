from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import src.hparams as model_config

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [ batch_size x n_heads x seq_len x hidden_size ]

        attn = q @ k.transpose(-1, -2)
        attn /= self.temperature

        if mask is not None:
            attn.masked_fill_(mask, -torch.inf)

        # attn: [ batch_size x n_heads x seq_len x seq_len ]
        attn = self.dropout(self.softmax(attn))

        # output: [ batch_size x n_heads x seq_len x hidden_size ]
        output = attn @ v
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_model = d_model

        self.QKV = nn.Linear(d_model, 3 * d_model)

        self.attention = ScaledDotProductAttention(
            temperature=self.d_k**0.5)  # TODO: fix
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_model, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # normal distribution initialization better than kaiming(default in pytorch)
        nn.init.normal_(self.QKV.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))

    def forward(self, x, mask=None):
        d_model, n_head = self.d_model, self.n_head

        B, T, C = x.size()

        residual = x
        # pre-norm
        x = self.layer_norm(x)

        q, k, v = self.QKV(x).split(d_model, dim=2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

        if mask is not None:
            mask = mask.repeat(1, n_head, 1, 1)  # b x n x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(B, T, C)

        output = self.dropout(self.fc(output))
        output = (output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=model_config.fft_conv1d_kernel[0], padding=model_config.fft_conv1d_padding[0])
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=model_config.fft_conv1d_kernel[1], padding=model_config.fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = output + residual
        return output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, mask=slf_attn_mask)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class VariancePredictor(nn.Module):
    """ Duration/pitch/energy Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_layer = nn.ModuleDict(OrderedDict([
            ("transpose1", Transpose(-1, -2)),
            ("conv1d_1", nn.Conv1d(self.input_size,
                                   self.filter_size,
                                   kernel_size=self.kernel,
                                   padding=1)),
            ("transpose2", Transpose(-1, -2)),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("relu_1", nn.ReLU()),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("transpose3", Transpose(-1, -2)),
            ("conv1d_2", nn.Conv1d(self.filter_size,
                                   self.filter_size,
                                   kernel_size=self.kernel,
                                   padding=1)),
            ("transpose4", Transpose(-1, -2)),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("relu_2", nn.ReLU()),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        for key, layer in self.conv_layer.items():
            encoder_output = layer(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = (
                duration_predictor_output*alpha + 0.5).int()
            output = self.LR(x, duration_predictor_output, mel_max_length)
            mel_pos = torch.stack([torch.tensor(
                [i+1 for i in range(output.size(1))], device=x.device, dtype=torch.int64)])
            return output, mel_pos


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, model_config, stats, offset=1.):
        super(VarianceAdaptor, self).__init__()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.offset = offset
        self.register_buffer("pitch_bins", torch.exp(
            torch.linspace(torch.log(stats["f0"]["min"]+offset), torch.log(stats["f0"]["max"]+offset), model_config.quantization_bins-1)))
        self.register_buffer("energy_bins", torch.exp(
            torch.linspace(torch.log(stats["energy"]["min"]+offset), torch.log(stats["energy"]["max"]+offset), model_config.quantization_bins-1)))
        self.pitch_embed = nn.Embedding(
            model_config.quantization_bins, model_config.encoder_dim)
        self.energy_embed = nn.Embedding(
            model_config.quantization_bins, model_config.encoder_dim)

    def forward(self, x, beta=1.0, gamma=1.0, pitch=None, energy=None):
        offset = self.offset
        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.pitch_predictor(x)
        if pitch is not None and energy is not None:
            pitch = torch.bucketize(pitch+offset, self.pitch_bins)
            energy = torch.bucketize(pitch+offset, self.energy_bins)
            x = x + self.pitch_embed(pitch)
            x = x + self.pitch_embed(energy)
            return x, pitch_pred, energy_pred
        elif pitch is None and energy is None:
            pitch = torch.bucketize(pitch_pred*beta+offset, self.pitch_bins)
            energy = torch.bucketize(energy_pred*gamma+offset, self.energy_bins)
            x = x + self.pitch_embed(pitch)
            x = x + self.pitch_embed(energy)
            return x
        else:
            raise Exception("Either both or none of energy, pitch")


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(model_config.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(model_config.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1).unsqueeze(1)  # b x 1 x lq x lk

    return padding_mask


class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()

        len_max_seq = model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.encoder_n_layer

        self.src_word_emb = nn.Embedding(
            model_config.vocab_size,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):

        super(Decoder, self).__init__()

        len_max_seq = model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.decoder_n_layer

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config, stats):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config, stats)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(
            model_config.decoder_dim, model_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None,
                pitch_target=None, energy_target=None, alpha=1.0, beta=1.0, gamma=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)

        if self.training:
            output, duration_predictor_output = self.length_regulator(
                x, alpha, length_target, mel_max_length)
            output, pitch_prediction, energy_prediction = self.variance_adaptor(
                output, beta=beta, gamma=gamma, pitch=pitch_target, energy=energy_target)
            output = self.decoder(output, mel_pos)

            output = self.mask_tensor(output, mel_pos, mel_max_length)

            output = self.mel_linear(output)
            return output, duration_predictor_output, pitch_prediction, energy_prediction
        else:
            output, mel_pos = self.length_regulator(x, alpha)
            output = self.variance_adaptor(
                self, output, beta=beta, gamma=gamma)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return output
