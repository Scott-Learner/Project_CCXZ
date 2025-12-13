"""
Test script for autoencoding checkpoint
Load ETTm1 data and pass through trained NormalizedWaveletAutoencoder model
"""
import torch
import argparse
import os
from data_provider.data_factory import data_provider
from exp.exp_autoencoding import NormalizedWaveletAutoencoder


def get_args():
    """Setup arguments for ETTm1"""
    args = argparse.Namespace(
        # Data config
        data='ETTm1',
        root_path='./data/ETT/',
        data_path='ETTm1.csv',
        features='M',
        target='OT',
        freq='t',
        
        # Autoencoding config (must match checkpoint)
        seq_len=512,
        level=1,
        archi='DWT',
        wavelet='db2',
        num_channels=7,
        lambda_l1=1.0,
        
        # Data loader config
        label_len=48,
        pred_len=96,
        embed='timeF',
        seasonal_patterns='Monthly',
        batch_size=8,
        num_workers=0,
        
        # Checkpoint path (simple format)
        checkpoint_dir='./checkpoints/autoencoding/ETTm1'
    )
    return args


def build_model(args):
    """Build NormalizedWaveletAutoencoder model"""
    model = NormalizedWaveletAutoencoder(
        seq_len=args.seq_len,
        level=args.level,
        archi=args.archi,
        wavelet=args.wavelet
    )
    return model


def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data
    print("\n" + "="*60)
    print("Loading ETTm1 data...")
    test_data, test_loader = data_provider(args, flag='test')
    
    # Get one batch
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
    batch_x = batch_x.float().to(device)
    print(f"Input shape: {batch_x.shape}")  # [batch, seq_len, channels]
    
    # 2. Load models for each channel
    print("\n" + "="*60)
    print("Loading trained models...")
    models = []
    for ch_idx in range(args.num_channels):
        model = build_model(args).to(device)
        ckpt_path = os.path.join(args.checkpoint_dir, f'channel_{ch_idx}_checkpoint.pth')
        
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"  ✓ Loaded channel {ch_idx}: {ckpt_path}")
        else:
            print(f"  ✗ Checkpoint not found: {ckpt_path}")
        
        model.eval()
        models.append(model)
    
    # 3. Process each channel through the model (with normalization)
    print("\n" + "="*60)
    print("Processing through NormalizedWaveletAutoencoder...")
    
    all_coeffs = []
    all_recons = []
    
    criterion = torch.nn.L1Loss()
    
    with torch.no_grad():
        for ch_idx in range(args.num_channels):
            # Extract single channel: [batch, seq_len] -> [batch, 1, seq_len]
            x_ch = batch_x[:, :, ch_idx].unsqueeze(1)
            
            # Use the model's encode method (includes normalization)
            Emb, means, stdev = models[ch_idx].encode(x_ch)
            
            # Use the model's decode method (includes denormalization)
            x_recon = models[ch_idx].decode(Emb, means, stdev)
            
            all_coeffs.append(Emb)
            all_recons.append(x_recon)
            
            # Print info for first channel
            if ch_idx == 0:
                print(f"\nChannel {ch_idx} details:")
                print(f"  Input shape: {x_ch.shape}")
                print(f"  Coefficients (Emb): {len(Emb)} levels")
                for i, coef in enumerate(Emb):
                    print(f"    Level {i}: {coef.shape}")
                print(f"  Reconstruction shape: {x_recon.shape}")
    
    # 4. Calculate reconstruction error
    print("\n" + "="*60)
    print("Reconstruction results:")
    
    total_mse = 0
    total_loss = 0
    for ch_idx in range(args.num_channels):
        x_ch = batch_x[:, :, ch_idx].unsqueeze(1)
        x_recon = all_recons[ch_idx]
        
        mse = torch.mean((x_ch - x_recon) ** 2).item()
        loss, _, _ = models[ch_idx].compute_loss(x_ch, criterion, args.lambda_l1)
        
        total_mse += mse
        total_loss += loss.item()
        print(f"  Channel {ch_idx}: MSE = {mse:.6f}, Loss = {loss.item():.6f}")
    
    print(f"\n  Average MSE: {total_mse / args.num_channels:.6f}")
    print(f"  Average Loss: {total_loss / args.num_channels:.6f}")
    
    # 5. Return outputs
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Input: {batch_x.shape}")
    print(f"  Channels processed: {args.num_channels}")
    print(f"  Wavelet: {args.wavelet}, Level: {args.level}")
    print("="*60)
    
    return all_coeffs, all_recons


if __name__ == '__main__':
    coeffs, recons = main()
    print("\n✅ Test completed!")
