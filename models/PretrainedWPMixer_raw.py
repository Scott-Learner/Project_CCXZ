"""
PretrainedWPMixer_raw: WPMixer with pretrained wavelet from serial training checkpoints.

This version is specifically for loading checkpoints from serial (per-channel) training,
where each channel is saved as: channel_{ch_idx}_checkpoint.pth

Key points:
1. Loads pretrained NormalizedWaveletAutoencoder weights for each channel independently
2. Compatible with checkpoints from exp_autoencoding_output.py (serial training)
3. The autoencoder was trained on normalized data, and forecast() also normalizes input
4. So the decomposition operates in normalized space - no double normalization needed
"""
import torch
import torch.nn as nn
import os
from layers.NeuralDWAV import NeuralDWAV
from models.WPMixer import ResolutionBranch


class Pretrained_LWPT_Decomposition(nn.Module):
    """
    Wavelet decomposition layer that loads pretrained weights from serial autoencoding checkpoints.
    
    Expected checkpoint format: {checkpoint_dir}/channel_{ch_idx}_checkpoint.pth
    
    The pretrained autoencoder (NormalizedWaveletAutoencoder) contains:
        - wavelet_model.Filt.kernel.*  (wavelet filter coefficients)
        - wavelet_model.Act.bias_p.*   (activation thresholds)
    
    Important: Input data should already be normalized before calling transform(),
    since the autoencoder was trained on normalized data.
    """
    def __init__(self,
                 input_length,
                 pred_length,
                 level,
                 channel,
                 device,
                 checkpoint_dir,
                 wavelet_name='db2',
                 freeze_wavelet=True):
        super(Pretrained_LWPT_Decomposition, self).__init__()
        
        self.input_length = input_length
        self.pred_length = pred_length
        self.level = level
        self.channel = channel
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.wavelet_name = wavelet_name
        self.freeze_wavelet = freeze_wavelet
        self.eps = 1e-5
        
        # Create NeuralDWAV for each channel and load pretrained weights
        self.ndwav_list = nn.ModuleList()
        
        for ch_idx in range(self.channel):
            # Create NeuralDWAV with same config as NormalizedWaveletAutoencoder
            ndwav = NeuralDWAV(
                Input_Size=self.input_length,
                Input_Level=self.level,
                Input_Archi="DWT",
                Filt_Trans=True,
                Filt_Train=True,
                Filt_Tfree=False,
                Filt_Style="Layer_Free",  # Must match autoencoder config
                Filt_Mother=self.wavelet_name,
                Act_Train=True,
                Act_Style="Sigmoid",
                Act_Symmetric=True,
                Act_Init=0
            ).to(self.device)
            
            # Load pretrained weights from individual channel checkpoint
            loaded = False
            ckpt_path = os.path.join(self.checkpoint_dir, f'channel_{ch_idx}_checkpoint.pth')
            
            if os.path.exists(ckpt_path):
                try:
                    state_dict = torch.load(ckpt_path, map_location=self.device)
                    
                    # Extract wavelet_model.* weights and rename to match NeuralDWAV
                    ndwav_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('wavelet_model.'):
                            new_key = key.replace('wavelet_model.', '')
                            ndwav_state_dict[new_key] = value
                    
                    if ndwav_state_dict:
                        ndwav.load_state_dict(ndwav_state_dict)
                        print(f"  ✓ Loaded pretrained wavelet for channel {ch_idx} from {ckpt_path}")
                        loaded = True
                    else:
                        print(f"  ⚠ No wavelet_model weights found in {ckpt_path}")
                except Exception as e:
                    print(f"  ⚠ Error loading checkpoint for channel {ch_idx}: {e}")
            
            if not loaded:
                print(f"  ⚠ Checkpoint not found: {ckpt_path}")
                print(f"    Using random initialization with {self.wavelet_name}")
            
            # Optionally freeze wavelet parameters
            if self.freeze_wavelet:
                for param in ndwav.parameters():
                    param.requires_grad = False
            
            self.ndwav_list.append(ndwav)
        
        # Compute coefficient dimensions
        self.input_w_dim = self._compute_coef_dims(self.input_length)
        self.pred_w_dim = self._scale_pred_dims(self.input_w_dim, self.input_length, self.pred_length)
        
        print(f"  Decomposition dims: input={self.input_w_dim}, pred={self.pred_w_dim}")
    
    def _compute_coef_dims(self, seq_len):
        """Compute coefficient dimensions by doing a dummy forward pass."""
        dummy_x = torch.zeros((1, 1, seq_len), device=self.device)
        embeddings = self.ndwav_list[0].LDWT(dummy_x)
        
        dims = []
        # Approximation (low-pass) coefficient
        dims.append(embeddings[self.level].shape[-1])
        # Detail coefficients
        for i in range(self.level):
            dims.append(embeddings[i].shape[-1])
        return dims
    
    def _scale_pred_dims(self, input_w_dim, input_length, pred_length):
        """Scale coefficient dimensions for prediction length."""
        ratio = pred_length / float(input_length)
        return [max(1, int(round(ratio * L))) for L in input_w_dim]
    
    def transform(self, x):
        """
        Wavelet decomposition.
        
        Args:
            x: [batch, channel, seq] - should be already normalized!
        
        Returns:
            yl: [batch, channel, L0] approximation coefficients
            yh: list of [batch, channel, Li] detail coefficients
        """
        B, C, T = x.shape
        
        all_yl = []
        all_yh = [[] for _ in range(self.level)]
        
        for c in range(C):
            # Extract single channel: [B, 1, T]
            x_c = x[:, c:c+1, :].contiguous()
            
            # Apply pretrained wavelet transform
            embeddings = self.ndwav_list[c].LDWT(x_c)
            
            # Approximation coefficient
            all_yl.append(embeddings[self.level])
            
            # Detail coefficients
            for i in range(self.level):
                all_yh[i].append(embeddings[i])
        
        # Concatenate across channels
        yl = torch.cat(all_yl, dim=1)  # [B, C, L0]
        yh = [torch.cat(all_yh[i], dim=1) for i in range(self.level)]
        
        return yl, yh
    
    def inv_transform(self, yl, yh):
        """
        Inverse wavelet transform.
        
        Args:
            yl: [batch, channel, L0] approximation coefficients
            yh: list of [batch, channel, Li] detail coefficients
        
        Returns:
            x: [batch, channel, seq] reconstructed signal
        """
        B, C, _ = yl.shape
        
        all_x = []
        
        for c in range(C):
            # Build embeddings list for NeuralDWAV.iLDWT
            embeddings = [None] * (self.level + 1)
            
            # Detail coefficients
            for i in range(self.level):
                embeddings[i] = yh[i][:, c:c+1, :].contiguous()
            
            # Approximation coefficient
            embeddings[self.level] = yl[:, c:c+1, :].contiguous()
            
            # Apply pretrained inverse transform
            x_c = self.ndwav_list[c].iLDWT(embeddings)
            all_x.append(x_c)
        
        x = torch.cat(all_x, dim=1)  # [B, C, T]
        return x


class WPMixerCore(nn.Module):
    """WPMixer core with pretrained wavelet decomposition."""
    
    def __init__(self,
                 input_length,
                 pred_length,
                 level,
                 batch_size,
                 channel,
                 d_model,
                 dropout,
                 embedding_dropout,
                 tfactor,
                 dfactor,
                 device,
                 patch_len,
                 patch_stride,
                 checkpoint_dir,
                 wavelet_name='db2',
                 freeze_wavelet=True):
        super(WPMixerCore, self).__init__()
        
        self.input_length = input_length
        self.pred_length = pred_length
        self.level = level
        self.channel = channel
        self.device = device
        
        print(f"\n{'='*60}")
        print(f"Initializing PretrainedWPMixer_raw (Serial Checkpoints)")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Channels: {channel}, Level: {level}")
        print(f"  Freeze wavelet: {freeze_wavelet}")
        print(f"  Expected format: channel_{{idx}}_checkpoint.pth")
        print(f"{'='*60}")
        
        # Pretrained wavelet decomposition
        self.Decomposition_model = Pretrained_LWPT_Decomposition(
            input_length=input_length,
            pred_length=pred_length,
            level=level,
            channel=channel,
            device=device,
            checkpoint_dir=checkpoint_dir,
            wavelet_name=wavelet_name,
            freeze_wavelet=freeze_wavelet
        )
        
        self.input_w_dim = self.Decomposition_model.input_w_dim
        self.pred_w_dim = self.Decomposition_model.pred_w_dim
        
        # Resolution branches for each wavelet band
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
        
        print(f"{'='*60}\n")
    
    def forward(self, xL):
        """
        Args:
            xL: [Batch, seq_len, channel] - should be normalized!
        
        Returns:
            xT: [Batch, pred_len, channel]
        """
        x = xL.transpose(1, 2)  # [batch, channel, seq_len]
        
        # Wavelet decomposition (using pretrained wavelets)
        with torch.no_grad():
            xA, xD = self.Decomposition_model.transform(x)
        
        # Predict approximation coefficients
        yA = torch.concat((xA, self.resolutionBranch[0](xA)), dim=2)[:, :, -self.input_w_dim[0]:]
        
        # Predict detail coefficients
        yD = []
        for i in range(len(xD)):
            yD_i = torch.concat((xD[i], self.resolutionBranch[i + 1](xD[i])), dim=2)[:, :, -self.input_w_dim[i+1]:]
            yD.append(yD_i)
        
        # Inverse wavelet transform
        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)
        
        xT = y[:, -self.pred_length:, :]
        return xT


class Model(nn.Module):
    """
    PretrainedWPMixer_raw: WPMixer with pretrained wavelet from serial training.
    
    This version loads checkpoints saved by serial (per-channel) training.
    Use this for ECL and other datasets trained with exp_autoencoding_output.py
    
    Args:
        args: Model arguments
        checkpoint_dir: Path to autoencoding checkpoints (e.g., './checkpoints/autoencoding/ECL/')
        tfactor, dfactor: Mixer factors
        wavelet: Wavelet name (must match pretrained model)
        level: Decomposition level (must match pretrained model)
        stride: Patch stride
        freeze_wavelet: Whether to freeze pretrained wavelet weights
    """
    
    def __init__(self, args, checkpoint_dir=None, tfactor=5, dfactor=5, 
                 wavelet=None, level=None, stride=None, freeze_wavelet=True):
        super(Model, self).__init__()
        self.args = args
        self.task_name = args.task_name
        # If explicit checkpoint_dir is not provided, fall back to args.pretrained_ckpt
        ckpt_dir = checkpoint_dir or getattr(args, 'pretrained_ckpt', './checkpoints/autoencoding')
        
        # Read wavelet parameters from args if not explicitly provided
        wavelet_name = wavelet or getattr(args, 'wavelet_name', 'db2')
        wavelet_level = level or getattr(args, 'wavelet_level', 1)
        patch_stride = stride or getattr(args, 'patch_stride', 8)
        
        self.wpmixerCore = WPMixerCore(
            input_length=self.args.seq_len,
            pred_length=self.args.pred_len,
            level=wavelet_level,
            batch_size=self.args.batch_size,
            channel=self.args.c_out,
            d_model=self.args.d_model,
            dropout=self.args.dropout,
            embedding_dropout=self.args.dropout,
            tfactor=tfactor,
            dfactor=dfactor,
            device=self.args.device,
            patch_len=self.args.patch_len,
            patch_stride=patch_stride,
            checkpoint_dir=ckpt_dir,
            wavelet_name=wavelet_name,
            freeze_wavelet=freeze_wavelet
        )
    
    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        """
        Forecasting with normalization.
        
        Note: Input is normalized here before passing to WPMixerCore.
        The pretrained wavelet was also trained on normalized data,
        so this is consistent - no double normalization.
        """
        # Normalization (same as autoencoder training)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Forward through WPMixer (operates in normalized space)
        pred = self.wpmixerCore(x_enc)
        pred = pred[:, :, -self.args.c_out:]
        
        # Check for NaN or Inf in predictions before denormalization
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("⚠️  Warning: NaN or Inf detected in model predictions before denormalization")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # De-Normalization (use pred.shape[1] instead of self.args.pred_len)
        pred_len = pred.shape[1]
        dec_out = pred * (stdev[:, 0].unsqueeze(1).repeat(1, pred_len, 1))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, pred_len, 1))
        
        # Final check
        if torch.isnan(dec_out).any() or torch.isinf(dec_out).any():
            print("⚠️  Warning: NaN or Inf detected after denormalization")
            dec_out = torch.nan_to_num(dec_out, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, L, D]
        
        raise NotImplementedError(f"Task {self.task_name} is not supported")

