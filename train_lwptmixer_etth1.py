"""
Simple training script for LWPTMixer on ETTh1 dataset
Uses NeuralDWAV for learnable wavelet decomposition
"""
import os
import sys

# Training configurations
configs = {
    'task_name': 'long_term_forecast',
    'is_training': 1,
    'root_path': './data/ETT/',
    'data_path': 'ETTh1.csv',
    'model': 'LWPTMixer',
    'data': 'ETTh1',
    'features': 'M',
    'seq_len': 512,
    'label_len': 48,
    'enc_in': 7,
    'dec_in': 7,
    'c_out': 7,
    'd_model': 256,
    'd_ff': 2048,
    'e_layers': 2,
    'd_layers': 1,
    'batch_size': 16,
    'learning_rate': 0.001,
    'train_epochs': 10,
    'patience': 3,
    'patch_len': 16,
    'des': 'NeuralDWAV',
    'itr': 1,
}

# Prediction lengths to test
pred_lens = [96, 192, 336, 720]

def run_training(pred_len):
    """Run training for a specific prediction length"""
    model_id = f"ETTh1_512_{pred_len}"
    
    cmd_parts = ['python', '-u', 'run.py']
    
    # Add all configuration parameters
    for key, value in configs.items():
        cmd_parts.append(f'--{key}')
        cmd_parts.append(str(value))
    
    # Add pred_len and model_id
    cmd_parts.extend(['--pred_len', str(pred_len)])
    cmd_parts.extend(['--model_id', model_id])
    
    # Join command
    cmd = ' '.join(cmd_parts)
    
    print("="*80)
    print(f"Training LWPTMixer for ETTh1 with prediction length {pred_len}")
    print("="*80)
    print(f"Command: {cmd}\n")
    
    # Execute
    os.system(cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train LWPTMixer on ETTh1')
    parser.add_argument('--quick', action='store_true', help='Quick test with pred_len=96 only')
    parser.add_argument('--pred_len', type=int, default=None, help='Specific prediction length')
    args = parser.parse_args()
    
    if args.pred_len:
        # Train for specific pred_len
        run_training(args.pred_len)
    elif args.quick:
        # Quick test: only pred_len=96, fewer epochs
        configs['train_epochs'] = 3
        configs['des'] = 'NeuralDWAV_Quick'
        run_training(96)
    else:
        # Full training: all prediction lengths
        for pred_len in pred_lens:
            run_training(pred_len)
            print(f"\n{'='*80}\n")

    print("\n" + "="*80)
    print("✅ Training completed!")
    print("="*80)


