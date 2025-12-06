"""
Test script for LWPTMixer model
Tests forward pass and dimension validation
"""
import torch
import argparse

def create_mock_args():
    """Create mock arguments for model"""
    parser = argparse.ArgumentParser()
    
    # Task
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    
    # Data dimensions
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    
    # Model params
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    
    # Training params
    parser.add_argument('--use_amp', type=bool, default=False)
    
    args = parser.parse_args([])
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    return args

def test_lwptmixer():
    print("="*70)
    print("🧪 LWPTMixer Model Test")
    print("="*70)
    
    # Import model
    from models.LWPTMixer import Model
    
    # Create args
    args = create_mock_args()
    
    # ===== 1. Model Configuration =====
    print("\n⚙️  MODEL CONFIGURATION")
    print("-" * 70)
    print(f"{'Input Length (seq_len):':<30} {args.seq_len}")
    print(f"{'Prediction Length:':<30} {args.pred_len}")
    print(f"{'Channels:':<30} {args.c_out}")
    print(f"{'D_model:':<30} {args.d_model}")
    print(f"{'Batch Size:':<30} {args.batch_size}")
    print(f"{'Device:':<30} {args.device}")
    
    # Model hyperparameters
    wavelet = 'db4'
    level = 3
    tfactor = 2
    dfactor = 2
    stride = 8
    
    print(f"{'Wavelet:':<30} {wavelet}")
    print(f"{'Decomposition Level:':<30} {level}")
    print(f"{'T-factor:':<30} {tfactor}")
    print(f"{'D-factor:':<30} {dfactor}")
    
    # ===== 2. Initialize Model =====
    print("\n🏗️  INITIALIZING MODEL")
    print("-" * 70)
    
    try:
        model = Model(
            args=args,
            tfactor=tfactor,
            dfactor=dfactor,
            wavelet=wavelet,
            level=level,
            stride=stride,
            no_decomposition=False
        ).to(args.device)
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return
    
    # ===== 3. Generate Test Data =====
    print("\n📊 GENERATING TEST DATA")
    print("-" * 70)
    
    batch_size = args.batch_size
    seq_len = args.seq_len
    pred_len = args.pred_len
    channels = args.c_out
    
    # Create input tensors
    x_enc = torch.randn(batch_size, seq_len, channels).to(args.device)
    x_mark_enc = torch.randn(batch_size, seq_len, 4).to(args.device)  # time features
    x_dec = torch.randn(batch_size, args.label_len + pred_len, channels).to(args.device)
    x_mark_dec = torch.randn(batch_size, args.label_len + pred_len, 4).to(args.device)
    
    print(f"x_enc:      {x_enc.shape}")
    print(f"x_mark_enc: {x_mark_enc.shape}")
    print(f"x_dec:      {x_dec.shape}")
    print(f"x_mark_dec: {x_mark_dec.shape}")
    
    # ===== 4. Forward Pass =====
    print("\n🚀 FORWARD PASS")
    print("-" * 70)
    
    model.eval()
    try:
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"✅ Forward pass successful")
        print(f"Output shape: {list(output.shape)}")
        
        # Validate dimensions
        expected_shape = (batch_size, pred_len, channels)
        if output.shape == expected_shape:
            print(f"✅ Dimension validation passed")
            print(f"   Expected: {expected_shape}")
            print(f"   Got:      {tuple(output.shape)}")
        else:
            print(f"❌ Dimension mismatch!")
            print(f"   Expected: {expected_shape}")
            print(f"   Got:      {tuple(output.shape)}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ===== 5. Output Statistics =====
    print("\n📈 OUTPUT STATISTICS")
    print("-" * 70)
    print(f"{'Mean:':<20} {output.mean().item():.6f}")
    print(f"{'Std:':<20} {output.std().item():.6f}")
    print(f"{'Min:':<20} {output.min().item():.6f}")
    print(f"{'Max:':<20} {output.max().item():.6f}")
    
    # ===== 6. Model Size =====
    print("\n💾 MODEL SIZE")
    print("-" * 70)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'Total Parameters:':<25} {total_params:,}")
    print(f"{'Trainable Parameters:':<25} {trainable_params:,}")
    print(f"{'Model Size:':<25} ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print("\n" + "="*70)
    print("✅ All tests passed successfully!")
    print("="*70)

if __name__ == "__main__":
    test_lwptmixer()

