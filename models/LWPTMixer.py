import torch.nn as nn
import torch
from layers.NeuralDWAV_Decomposition import LWPT_Decomposition
from models.WPMixer import Mixer, TokenMixer, ResolutionBranch

class WPMixerCore(nn.Module):
    def __init__(self,
                 input_length=[],
                 pred_length=[],
                 wavelet_name=[],
                 level=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 device=[],
                 patch_len=[],
                 patch_stride=[],
                 no_decomposition=[],
                 use_amp=[],
                 per_channel_wavelet=True):
        super(WPMixerCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp
        self.per_channel_wavelet = per_channel_wavelet

        self.Decomposition_model = LWPT_Decomposition(input_length=self.input_length,
                                                 pred_length=self.pred_length,
                                                 wavelet_name=self.wavelet_name,
                                                 level=self.level,
                                                 batch_size=self.batch_size,
                                                 channel=self.channel,
                                                 d_model=self.d_model,
                                                 tfactor=self.tfactor,
                                                 dfactor=self.dfactor,
                                                 device=self.device,
                                                 no_decomposition=self.no_decomposition,
                                                 use_amp=self.use_amp,
                                                 per_channel_wavelet=self.per_channel_wavelet)

        self.input_w_dim = self.Decomposition_model.input_w_dim  # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim  # list of the length of the predicted coefficient series

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # (m+1) number of resolutionBranch
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq=self.input_w_dim[i],
                                                                pred_seq=self.pred_w_dim[i],
                                                                batch_size=self.batch_size,
                                                                channel=self.channel,
                                                                d_model=self.d_model,
                                                                dropout=self.dropout,
                                                                embedding_dropout=self.embedding_dropout,
                                                                tfactor=self.tfactor,
                                                                dfactor=self.dfactor,
                                                                patch_len=self.patch_len,
                                                                patch_stride=self.patch_stride) for i in
                                               range(len(self.input_w_dim))])

    def forward(self, xL):
        '''
        Parameters
        ----------
        xL : Look back window: [Batch, look_back_length, channel]

        Returns
        -------
        xT : Prediction time series: [Batch, prediction_length, output_channel]
        '''
        x = xL.transpose(1, 2)  # [batch, channel, look_back_length]

        # xA: approximation coefficient series,
        # xD: detail coefficient series
        # yA: predicted approximation coefficient series
        # yD: predicted detail coefficient series

        xA, xD = self.Decomposition_model.transform(x)

        yA = torch.concat((xA, self.resolutionBranch[0](xA)), dim=2)[:, :, -self.input_w_dim[0]:]
        yD = []
        for i in range(len(xD)):
            yD_i = torch.concat((xD[i], self.resolutionBranch[i + 1](xD[i])), dim=2)[:, :, -self.input_w_dim[i+1]:]
            yD.append(yD_i)

        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)
        xT = y[:, -self.pred_length:, :]  # decomposition output is always even, but pred length can be odd

        return xT


class Model(nn.Module):
    def __init__(self, args, tfactor=5, dfactor=5, wavelet='db2', level=3, stride=8, 
                 no_decomposition=False, per_channel_wavelet=True):
        super(Model, self).__init__()
        self.args = args
        self.task_name = args.task_name
        self.wpmixerCore = WPMixerCore(input_length=self.args.seq_len,
                                       pred_length=self.args.pred_len,
                                       wavelet_name=wavelet,
                                       level=level,
                                       batch_size=self.args.batch_size,
                                       channel=self.args.c_out,
                                       d_model=self.args.d_model,
                                       dropout=self.args.dropout,
                                       embedding_dropout=self.args.dropout,
                                       tfactor=tfactor,
                                       dfactor=dfactor,
                                       device=self.args.device,
                                       patch_len=self.args.patch_len,
                                       patch_stride=stride,
                                       no_decomposition=no_decomposition,
                                       use_amp=self.args.use_amp,
                                       per_channel_wavelet=per_channel_wavelet)

    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        pred = self.wpmixerCore(x_enc)
        pred = pred[:, :, -self.args.c_out:]

        # De-Normalization
        dec_out = pred * (stdev[:, 0].unsqueeze(1).repeat(1, self.args.pred_len, 1))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.args.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, L, D]
        if self.task_name == 'imputation':
            raise NotImplementedError("Task imputation for WPMixer is temporarily not supported")
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError("Task anomaly_detection for WPMixer is temporarily not supported")
        if self.task_name == 'classification':
            raise NotImplementedError("Task classification for WPMixer is temporarily not supported")
        return None
