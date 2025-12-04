#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test to verify MultiWaveletDecomposition dimension alignment works correctly.
"""

import torch
import sys
sys.path.insert(0, '/workspace/Project_CCXZ')

from layers.MultiWaveletDecomposition import MultiWaveletDecomposition

def test_dimension_alignment():
    """Test that dimension alignment works for multiple wavelets."""
    
    print("=" * 80)
    print("Testing MultiWaveletDecomposition with dimension alignment")
    print("=" * 80)
    
    # Test parameters
    batch_size = 4
    channel = 7
    input_length = 512
    pred_length = 96
    
    # Use different wavelets with different filter lengths
    wavelet_names = ['db2', 'db3', 'db4']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create decomposition module
    print(f"\nCreating MultiWaveletDecomposition with wavelets: {wavelet_names}")
    decomp = MultiWaveletDecomposition(
        input_length=input_length,
        pred_length=pred_length,
        wavelet_names=wavelet_names,
        wavelet_weights=None,  # Equal weights
        level=1,
        batch_size=batch_size,
        channel=channel,
        device=device,
        learnable_weights=True,
        use_amp=False
    )
    
    if device.type == 'cuda':
        decomp = decomp.cuda()
    
    print(f"Input dimensions after alignment: {decomp.input_w_dim}")
    print(f"Prediction dimensions after alignment: {decomp.pred_w_dim}")
    
    # Create test input
    print(f"\nCreating test input: shape ({batch_size}, {channel}, {input_length})")
    x = torch.randn(batch_size, channel, input_length)
    if device.type == 'cuda':
        x = x.cuda()
    
    # Test forward transform
    print("\nPerforming forward transform...")
    try:
        yl, yh = decomp.transform(x)
        print(f"✓ Forward transform successful!")
        print(f"  yl shape: {yl.shape}")
        for i, yh_level in enumerate(yh):
            print(f"  yh[{i}] shape: {yh_level.shape}")
    except Exception as e:
        print(f"✗ Forward transform failed: {e}")
        return False
    
    # Test inverse transform
    print("\nPerforming inverse transform...")
    try:
        x_reconstructed = decomp.inv_transform(yl, yh)
        print(f"✓ Inverse transform successful!")
        print(f"  Reconstructed shape: {x_reconstructed.shape}")
    except Exception as e:
        print(f"✗ Inverse transform failed: {e}")
        return False
    
    # Test with different input size
    print(f"\nTesting with prediction length input: shape ({batch_size}, {channel}, {pred_length})")
    x_pred = torch.randn(batch_size, channel, pred_length)
    if device.type == 'cuda':
        x_pred = x_pred.cuda()
    
    try:
        yl_pred, yh_pred = decomp.transform(x_pred)
        print(f"✓ Forward transform successful!")
        print(f"  yl shape: {yl_pred.shape}")
        for i, yh_level in enumerate(yh_pred):
            print(f"  yh[{i}] shape: {yh_level.shape}")
            
        x_pred_reconstructed = decomp.inv_transform(yl_pred, yh_pred)
        print(f"✓ Inverse transform successful!")
        print(f"  Reconstructed shape: {x_pred_reconstructed.shape}")
    except Exception as e:
        print(f"✗ Transform failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True

if __name__ == '__main__':
    success = test_dimension_alignment()
    sys.exit(0 if success else 1)

