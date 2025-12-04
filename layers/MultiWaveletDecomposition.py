# -*- coding: utf-8 -*-
"""
Multi-Wavelet Decomposition Module
Supports multiple wavelet bases with weighted combination and automatic dimension alignment.

Author: Based on WPMixer by Murad (SISLab, USF)
Enhanced with multi-wavelet support and dimension alignment
"""

import torch
import torch.nn as nn
from layers.DWT_Decomposition import DWT1DForward, DWT1DInverse


class MultiWaveletDecomposition(nn.Module):
    """
    Multi-Wavelet Decomposition with weighted combination and automatic dimension alignment.
    
    Uses multiple wavelet bases and combines their decomposition results with learnable 
    or fixed weights. Automatically aligns dimensions by cropping to minimum sizes across 
    all wavelets to ensure compatibility.
    
    Args:
        input_length: Length of input sequence
        pred_length: Length of prediction sequence
        wavelet_names: List of wavelet names, e.g., ['db2', 'db3', 'db4']
        wavelet_weights: List of weights for each wavelet (should sum to 1)
                        If None, uses equal weights
        level: Decomposition level
        batch_size: Batch size
        channel: Number of channels
        device: Device to use
        learnable_weights: If True, weights are learnable parameters
        use_amp: Use automatic mixed precision
    """
    
    def __init__(self,
                 input_length,
                 pred_length,
                 wavelet_names=['db2', 'db3', 'db4'],
                 wavelet_weights=None,
                 level=1,
                 batch_size=32,
                 channel=7,
                 device=None,
                 learnable_weights=True,
                 use_amp=False):
        super(MultiWaveletDecomposition, self).__init__()
        
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_names = wavelet_names
        self.num_wavelets = len(wavelet_names)
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.device = device
        self.use_amp = use_amp
        self.learnable_weights = learnable_weights
        
        # Initialize weights
        if wavelet_weights is None:
            # Equal weights by default
            wavelet_weights = [1.0 / self.num_wavelets] * self.num_wavelets
        
        assert len(wavelet_weights) == self.num_wavelets, \
            f"Number of weights ({len(wavelet_weights)}) must match number of wavelets ({self.num_wavelets})"
        
        # Normalize weights to sum to 1
        weight_sum = sum(wavelet_weights)
        wavelet_weights = [w / weight_sum for w in wavelet_weights]
        
        if learnable_weights:
            # Learnable weights (using softmax for normalization)
            self.weight_logits = nn.Parameter(
                torch.tensor([w for w in wavelet_weights], dtype=torch.float32)
            )
        else:
            # Fixed weights
            self.register_buffer(
                'weights',
                torch.tensor(wavelet_weights, dtype=torch.float32)
            )
        
        # Create DWT forward and inverse transforms for each wavelet
        self.dwt_list = nn.ModuleList()
        self.idwt_list = nn.ModuleList()
        
        for wavelet_name in wavelet_names:
            dwt = DWT1DForward(wave=wavelet_name, J=level, use_amp=use_amp)
            idwt = DWT1DInverse(wave=wavelet_name, use_amp=use_amp)
            
            if device is not None and device.type == 'cuda':
                dwt = dwt.cuda()
                idwt = idwt.cuda()
            
            self.dwt_list.append(dwt)
            self.idwt_list.append(idwt)
        
        # Compute output dimensions using minimum dimensions across all wavelets
        self.input_w_dim = self._dummy_forward(input_length)
        self.pred_w_dim = self._dummy_forward(pred_length)
    
    def get_weights(self):
        """Get normalized weights for each wavelet."""
        if self.learnable_weights:
            # Apply softmax to get normalized weights
            return torch.softmax(self.weight_logits, dim=0)
        else:
            return self.weights
    
    def _dummy_forward(self, input_length):
        """
        Compute output dimensions for decomposition using minimum dimensions across all wavelets.
        This ensures dimension compatibility when combining multiple wavelets.
        Returns minimum dimensions across ALL axes (batch, channel, seq_len).
        """
        dummy_x = torch.ones((self.batch_size, self.channel, input_length))
        if self.device is not None:
            dummy_x = dummy_x.to(self.device)
        
        # Collect all outputs from all wavelets
        all_yl = []
        all_yh = []
        
        for dwt in self.dwt_list:
            yl, yh = dwt(dummy_x)
            all_yl.append(yl)
            all_yh.append(yh)
        
        # Compute minimum dimensions across all wavelets for yl (all axes)
        min_yl_shape = [min(yl.shape[dim] for yl in all_yl) for dim in range(len(all_yl[0].shape))]
        
        # Compute minimum dimensions for yh at each level (all axes)
        min_yh_shapes = []
        for j in range(len(all_yh[0])):  # For each detail level
            min_shape = [min(all_yh[i][j].shape[dim] for i in range(len(all_yh))) 
                        for dim in range(len(all_yh[0][j].shape))]
            min_yh_shapes.append(min_shape)
        
        # Return as list: [yl_seq_dim, yh[0]_seq_dim, yh[1]_seq_dim, ...]
        # For backward compatibility, return only sequence length dimensions
        dims = [min_yl_shape[-1]] + [shape[-1] for shape in min_yh_shapes]
        return dims
    
    def transform(self, x):
        """
        Multi-wavelet forward transform with weighted combination and dimension alignment.
        
        Args:
            x: Input tensor of shape (batch, channel, seq_len)
            
        Returns:
            yl_combined: Weighted combination of approximation coefficients
            yh_combined: List of weighted combination of detail coefficients
        """
        weights = self.get_weights()
        
        # First pass: collect all outputs
        all_yl = []
        all_yh = []
        
        for i, dwt in enumerate(self.dwt_list):
            yl, yh = dwt(x)
            all_yl.append(yl)
            all_yh.append(yh)
        
        # Find minimum dimensions across all wavelets for ALL dimensions
        # Assume shape is (batch, channel, seq_len)
        min_yl_shape = [min(yl.shape[dim] for yl in all_yl) for dim in range(len(all_yl[0].shape))]
        
        # For detail coefficients, find minimum for each level across all dimensions
        min_yh_shapes = []
        for j in range(len(all_yh[0])):  # For each detail level
            min_shape = [min(all_yh[i][j].shape[dim] for i in range(len(all_yh))) 
                        for dim in range(len(all_yh[0][j].shape))]
            min_yh_shapes.append(min_shape)
        
        # Second pass: combine with alignment by cropping to minimum dimensions
        yl_combined = None
        yh_combined = None
        
        for i, (weight, yl, yh) in enumerate(zip(weights, all_yl, all_yh)):
            # Crop approximation coefficients to minimum shape on all dimensions
            yl_cropped = yl[:min_yl_shape[0], :min_yl_shape[1], :min_yl_shape[2]]
            
            # Crop detail coefficients to minimum shapes on all dimensions
            yh_cropped = [yh[j][:min_yh_shapes[j][0], :min_yh_shapes[j][1], :min_yh_shapes[j][2]] 
                         for j in range(len(yh))]
            
            if yl_combined is None:
                # Initialize with first wavelet's output
                yl_combined = weight * yl_cropped
                yh_combined = [weight * yh_level for yh_level in yh_cropped]
            else:
                # Add weighted contribution
                yl_combined = yl_combined + weight * yl_cropped
                for j in range(len(yh)):
                    yh_combined[j] = yh_combined[j] + weight * yh_cropped[j]
        
        return yl_combined, yh_combined

    def inv_transform(self, yl, yh):
        """
        Multi-wavelet inverse transform with weighted combination and dimension alignment.

        Args:
            yl: Approximation coefficients
            yh: List of detail coefficients
            
        Returns:
            x_combined: Weighted combination of reconstructed signals
        """
        weights = self.get_weights()

        # First pass: collect all reconstructions
        all_x = []

        for i, idwt in enumerate(self.idwt_list):
            x_reconstructed = idwt((yl, yh))
            all_x.append(x_reconstructed)

        # Find minimum dimensions across all reconstructions for ALL dimensions
        min_x_shape = [min(x.shape[dim] for x in all_x) for dim in range(len(all_x[0].shape))]

        # Second pass: combine with alignment by cropping to minimum dimensions
        x_combined = None

        for i, (weight, x_reconstructed) in enumerate(zip(weights, all_x)):
            # Crop to minimum shape on all dimensions
            x_cropped = x_reconstructed[:min_x_shape[0], :min_x_shape[1], :min_x_shape[2]]
            
            if x_combined is None:
                x_combined = weight * x_cropped
            else:
                x_combined = x_combined + weight * x_cropped

        return x_combined

