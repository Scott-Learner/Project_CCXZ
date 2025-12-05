"""
Simple test script for NeuralDWAV Decomposition layer
"""
import torch
import sys
sys.path.append('./layers')

from NeuralDWAV_Decomposition import Decomposition

def test_neuraldwav_decomposition():
    # Simulation parameters
    batch_size = 8
    channel = 7
    input_length = 512
    pred_length = 96
    level = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("Testing NeuralDWAV Decomposition Layer")
    print("="*60)
    
    # Initialize decomposition layer
    decomp = Decomposition(
        input_length=input_length,
        pred_length=pred_length,
        wavelet_name='db4',
        level=level,
        batch_size=batch_size,
        channel=channel,
        d_model=256,
        tfactor=2,
        dfactor=2,
        device=device,
        no_decomposition=False,
        use_amp=False
    ).to(device).float()  # Ensure all parameters are float32
    
    # Create simulated input data: [batch, channel, seq_length]
    x = torch.randn(batch_size, channel, input_length).to(device)
    
    print(f"\n📥 Input shape: {list(x.shape)}")
    print(f"   [batch={batch_size}, channel={channel}, seq_length={input_length}]")
    
    # Forward transform (decomposition)
    with torch.no_grad():
        yl, yh = decomp.transform(x)
    
    print(f"\n📤 Output shapes after decomposition:")
    print(f"   Low-pass (yl):  {list(yl.shape)}")
    print(f"   High-pass (yh): {len(yh)} detail bands")
    for i, detail in enumerate(yh):
        print(f"      Level {i+1}: {list(detail.shape)}")
    
    # Inverse transform (reconstruction)
    with torch.no_grad():
        x_recon = decomp.inv_transform(yl, yh)
    
    print(f"\n🔄 Reconstructed shape: {list(x_recon.shape)}")
    
    # Check reconstruction error
    recon_error = torch.mean((x - x_recon[:, :, :input_length]) ** 2).item()
    print(f"\n📊 Reconstruction MSE: {recon_error:.6f}")
    
    # Show dimension information
    print(f"\n📐 Band dimensions:")
    print(f"   Input bands:  {decomp.input_w_dim}")
    print(f"   Pred bands:   {decomp.pred_w_dim}")
    
    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print("="*60)

if __name__ == "__main__":
    test_neuraldwav_decomposition()

