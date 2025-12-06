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
    level = 4
    wavelet_name = 'db4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("🔬 NeuralDWAV Decomposition Layer Test")
    print("="*70)
    
    # ===== 1. 显示超参数 =====
    print("\n⚙️  HYPERPARAMETERS")
    print("-" * 70)
    print(f"{'Wavelet Type:':<25} {wavelet_name}")
    print(f"{'Decomposition Level:':<25} {level}")
    print(f"{'Input Length:':<25} {input_length}")
    print(f"{'Prediction Length:':<25} {pred_length}")
    print(f"{'Batch Size:':<25} {batch_size}")
    print(f"{'Channels:':<25} {channel}")
    print(f"{'Device:':<25} {device}")
    
    # Initialize decomposition layer
    decomp = Decomposition(
        input_length=input_length,
        pred_length=pred_length,
        wavelet_name=wavelet_name,
        level=level,
        batch_size=batch_size,
        channel=channel,
        d_model=256,
        tfactor=2,
        dfactor=2,
        device=device,
        no_decomposition=False,
        use_amp=False
    ).to(device).float()
    
    # ===== 2. 显示小波核信息 =====
    print("\n🌊 WAVELET KERNEL INFO")
    print("-" * 70)
    if hasattr(decomp, 'ndwav') and hasattr(decomp.ndwav, 'Filt'):
        filt = decomp.ndwav.Filt
        print(f"{'Architecture:':<25} {decomp.ndwav.Archi}")
        print(f"{'Filter Type:':<25} Layer_Free (Learnable)")
        print(f"{'Kernel Length:':<25} {filt.lK}")
        print(f"{'Total Kernels:':<25} {len(filt.kernel)}")
        print(f"{'Trainable Params:':<25} {sum(p.numel() for p in filt.kernel if p.requires_grad)}")
        
        # 显示第一个核的初始值（前5个）
        kernel_0 = filt.kernel[0].detach().cpu().flatten()[:5]
        print(f"{'First Kernel (5 vals):':<25} [{', '.join([f'{v:.3f}' for v in kernel_0])}...]")
        
        # 显示 layerSize
        if hasattr(filt, 'layerSize'):
            print(f"{'Layer Sizes:':<25} {filt.layerSize}")
    
    # ===== 3. 显示维度信息 =====
    print("\n📐 DIMENSION INFO")
    print("-" * 70)
    print(f"{'Input Wavelet Dims:':<25} {decomp.input_w_dim}")
    print(f"{'Pred Wavelet Dims:':<25} {decomp.pred_w_dim}")
    print(f"{'Dimension Ratio:':<25} {pred_length / input_length:.4f}")
    
    # ===== 4. 前向测试 =====
    print("\n🔄 FORWARD PASS TEST")
    print("-" * 70)
    x = torch.randn(batch_size, channel, input_length).to(device)
    print(f"Input:  {list(x.shape)} [Batch, Channel, Seq]")
    
    with torch.no_grad():
        yl, yh = decomp.transform(x)
    
    print(f"Output: Low-pass (yl) = {list(yl.shape)}")
    for i, detail in enumerate(yh):
        print(f"        High-pass[{i}] (yh) = {list(detail.shape)}")
    
    # ===== 5. 逆变换测试 =====
    print("\n⏪ INVERSE TRANSFORM TEST")
    print("-" * 70)
    with torch.no_grad():
        x_recon = decomp.inv_transform(yl, yh)
    
    print(f"Reconstructed: {list(x_recon.shape)}")
    recon_error = torch.mean((x - x_recon[:, :, :input_length]) ** 2).item()
    print(f"MSE Error:     {recon_error:.6f}")
    
    if recon_error < 0.01:
        print(f"Status:        ✅ Excellent reconstruction")
    elif recon_error < 0.1:
        print(f"Status:        ✅ Good reconstruction")
    else:
        print(f"Status:        ⚠️  High error")
    
    # ===== 6. 统计信息 =====
    print("\n📊 STATISTICS")
    print("-" * 70)
    total_params = sum(p.numel() for p in decomp.parameters())
    trainable_params = sum(p.numel() for p in decomp.parameters() if p.requires_grad)
    print(f"{'Total Parameters:':<25} {total_params:,}")
    print(f"{'Trainable Parameters:':<25} {trainable_params:,}")
    print(f"{'Memory Usage:':<25} ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print("="*70)

if __name__ == "__main__":
    test_neuraldwav_decomposition()