# -*- coding: utf-8 -*-
"""
WPMixer with Multi-Wavelet Weighted Decomposition
Based on WPMixer by Murad (SISLab, USF)

This version supports multiple wavelets with weighted combination
for the decomposition results.
"""

import torch
import torch.nn as nn
from layers.MultiWaveletDecomposition import MultiWaveletDecomposition

class TokenMixer(nn.Module):
    def __init__(self, input_seq, batch_size, channel, pred_seq, dropout, factor, d_model):
        super(TokenMixer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_seq, pred_seq * factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_seq * factor, pred_seq)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x


class Mixer(nn.Module):
    def __init__(self, input_seq, out_seq, batch_size, channel, d_model, dropout, tfactor, dfactor):
        super(Mixer, self).__init__()
        self.tMixer = TokenMixer(
            input_seq=input_seq, batch_size=batch_size, channel=channel,
            pred_seq=out_seq, dropout=dropout, factor=tfactor, d_model=d_model
        )
        self.dropoutLayer = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm2d(channel)
        self.norm2 = nn.BatchNorm2d(channel)
        self.embeddingMixer = nn.Sequential(
            nn.Linear(d_model, d_model * dfactor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * dfactor, d_model)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dropoutLayer(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x))
        return x


class ResolutionBranch(nn.Module):
    def __init__(self, input_seq, pred_seq, batch_size, channel, d_model, 
                 dropout, embedding_dropout, tfactor, dfactor, patch_len, patch_stride):
        super(ResolutionBranch, self).__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = int((input_seq - patch_len) / patch_stride + 2)

        self.patch_norm = nn.BatchNorm2d(channel)
        self.patch_embedding_layer = nn.Linear(patch_len, d_model)
        self.mixer1 = Mixer(
            input_seq=self.patch_num, out_seq=self.patch_num,
            batch_size=batch_size, channel=channel, d_model=d_model,
            dropout=dropout, tfactor=tfactor, dfactor=dfactor
        )
        self.mixer2 = Mixer(
            input_seq=self.patch_num, out_seq=self.patch_num,
            batch_size=batch_size, channel=channel, d_model=d_model,
            dropout=dropout, tfactor=tfactor, dfactor=dfactor
        )
        self.norm = nn.BatchNorm2d(channel)
        self.dropoutLayer = nn.Dropout(embedding_dropout)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(self.patch_num * d_model, pred_seq)
        )

    def forward(self, x):
        x_patch = self.do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))
        out = self.mixer1(x_emb)
        res = out
        out = res + self.mixer2(out)
        out = self.norm(out)
        out = self.head(out)
        return out

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch


class WPMixerMultiWaveletCore(nn.Module):
    """
    WPMixer Core with Multi-Wavelet Decomposition.
    
    Args:
        wavelet_names: List of wavelet names for multi-wavelet decomposition
        wavelet_weights: Weights for each wavelet (default: equal weights)
        learnable_weights: If True, weights become learnable parameters
    """
    
    def __init__(self,
                 input_length,
                 pred_length,
                 wavelet_names=['db2', 'db3', 'db4'],
                 wavelet_weights=None,
                 level=1,
                 batch_size=32,
                 channel=7,
                 d_model=256,
                 dropout=0.1,
                 embedding_dropout=0.1,
                 tfactor=5,
                 dfactor=5,
                 device=None,
                 patch_len=16,
                 patch_stride=8,
                 learnable_weights=False,
                 use_amp=False):
        super(WPMixerMultiWaveletCore, self).__init__()
        
        self.input_length = input_length
        self.pred_length = pred_length
        self.channel = channel
        
        # Multi-Wavelet Decomposition
        self.Decomposition_model = MultiWaveletDecomposition(
            input_length=input_length,
            pred_length=pred_length,
            wavelet_names=wavelet_names,
            wavelet_weights=wavelet_weights,
            level=level,
            batch_size=batch_size,
            channel=channel,
            device=device,
            learnable_weights=learnable_weights,
            use_amp=use_amp
        )
        
        self.input_w_dim = self.Decomposition_model.input_w_dim
        self.pred_w_dim = self.Decomposition_model.pred_w_dim
        
        # Resolution branches for each decomposition level
        self.resolutionBranch = nn.ModuleList([
            ResolutionBranch(
                input_seq=self.input_w_dim[i],
                pred_seq=self.pred_w_dim[i],
                batch_size=batch_size,
                channel=channel,
                d_model=d_model,
                dropout=dropout,
                embedding_dropout=embedding_dropout,
                tfactor=tfactor,
                dfactor=dfactor,
                patch_len=patch_len,
                patch_stride=patch_stride
            ) for i in range(len(self.input_w_dim))
        ])

    def forward(self, xL):
        import torch.nn.functional as F
        
        x = xL.transpose(1, 2)  # [batch, channel, look_back_length]
        
        # Multi-wavelet decomposition
        xA, xD = self.Decomposition_model.transform(x)
        
        # Process approximation coefficients
        yA = self.resolutionBranch[0](xA)
        
        # Process detail coefficients
        yD = []
        for i in range(len(xD)):
            yD_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)
        
        # Multi-wavelet inverse transform
        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)  # [batch, seq_len, channel]
        
        # Get the actual output length after wavelet transform
        actual_len = y.shape[1]
        
        # If output length doesn't match pred_length due to wavelet transform,
        # use interpolation to adjust to target length
        if actual_len != self.pred_length:
            # Transpose to [batch, channel, seq_len] for interpolation
            y = y.transpose(1, 2)
            y = F.interpolate(y, size=self.pred_length, mode='linear', align_corners=False)
            y = y.transpose(1, 2)  # Back to [batch, seq_len, channel]
        
        xT = y[:, -self.pred_length:, :]
        
        return xT


class Model(nn.Module):
    """
    WPMixer Model with Multi-Wavelet Decomposition.
    
    Example usage:
        model = Model(args, 
                      wavelet_names=['db2', 'db3', 'db4'],
                      wavelet_weights=[0.5, 0.3, 0.2],
                      learnable_weights=False)
    """
    
    def __init__(self, args, 
                 tfactor=5, dfactor=5, 
                 wavelet_names=['db2', 'db3', 'db4'],
                 wavelet_weights=None,
                 level=1, stride=8, 
                 learnable_weights=False):
        super(Model, self).__init__()
        self.args = args
        self.task_name = args.task_name
        self.pred_length = args.pred_len
        
        self.wpmixerCore = WPMixerMultiWaveletCore(
            input_length=args.seq_len,
            pred_length=args.pred_len,
            wavelet_names=wavelet_names,
            wavelet_weights=wavelet_weights,
            level=level,
            batch_size=args.batch_size,
            channel=args.c_out,
            d_model=args.d_model,
            dropout=args.dropout,
            embedding_dropout=args.dropout,
            tfactor=tfactor,
            dfactor=dfactor,
            device=args.device,
            patch_len=args.patch_len,
            patch_stride=stride,
            learnable_weights=learnable_weights,
            use_amp=args.use_amp
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        pred = self.wpmixerCore(x_enc)
        pred = pred[:, :, -self.args.c_out:]

        # De-Normalization
        # Use actual prediction length from pred tensor (may differ from target due to wavelet transform)
        actual_pred_len = pred.shape[1]
        dec_out = pred * (stdev[:, 0].unsqueeze(1).repeat(1, actual_pred_len, 1))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, actual_pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            raise NotImplementedError("Task imputation for WPMixer is temporarily not supported")
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError("Task anomaly_detection for WPMixer is temporarily not supported")
        if self.task_name == 'classification':
            raise NotImplementedError("Task classification for WPMixer is temporarily not supported")
        return None
    
    def get_wavelet_weights(self):
        """Get current wavelet weights (useful for inspection/logging)."""
        return self.wpmixerCore.Decomposition_model.get_weights()

