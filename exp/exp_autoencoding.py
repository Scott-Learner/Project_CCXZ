from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import matplotlib.pyplot as plt
from layers.NeuralDWAV import NeuralDWAV

warnings.filterwarnings('ignore')

class NormalizedWaveletAutoencoder(nn.Module):
    """
    Wrapper for NeuralDWAV with built-in normalization and denormalization.
    """
    def __init__(self, seq_len, level, archi, wavelet):
        super(NormalizedWaveletAutoencoder, self).__init__()
        
        self.wavelet_model = NeuralDWAV(
            Input_Size=seq_len,
            Input_Level=level,
            Input_Archi=archi,
            Filt_Trans=True,
            Filt_Train=True,
            Filt_Tfree=False,
            Filt_Style="Layer_Free",
            Filt_Mother=wavelet,
            Act_Train=True,
            Act_Style="Sigmoid",
            Act_Symmetric=True,
            Act_Init=0
        ).float()
        
    def normalize(self, x):
        """Normalize input to zero mean and unit variance."""
        means = x.mean(dim=2, keepdim=True).detach()
        x_centered = x - means
        stdev = torch.sqrt(torch.var(x_centered, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_centered / stdev
        return x_norm, means, stdev
    
    def denormalize(self, x, means, stdev):
        """Denormalize output back to original scale."""
        return x * stdev + means
    
    def forward(self, x):
        """Full forward pass: normalize -> wavelet transform -> inverse transform -> denormalize"""
        x_norm, means, stdev = self.normalize(x)
        Emb = self.wavelet_model.T(x_norm)
        recon_norm = self.wavelet_model.iT(Emb)
        recon = self.denormalize(recon_norm, means, stdev)
        return recon
    
    def compute_loss(self, x, criterion, lambda_l1):
        """Compute reconstruction loss + sparsity loss."""
        x_norm, means, stdev = self.normalize(x)
        Emb = self.wavelet_model.T(x_norm)
        recon_norm = self.wavelet_model.iT(Emb)
        
        loss_recon = criterion(recon_norm, x_norm)
        loss_sparse = self.wavelet_model.L1_sum(Emb)
        total_loss = loss_recon + lambda_l1 * loss_sparse
        
        return total_loss, loss_recon, loss_sparse
    
    def L1_sum(self, Emb):
        return self.wavelet_model.L1_sum(Emb)


class Exp_Autoencoding(Exp_Basic):
    """
    Experiment class for autoencoding with NeuralDWAV.
    Implements parallelized per-channel training using torch.func.vmap.
    """
    
    def __init__(self, args):
        self.args = args
        
        # Set default values if not provided
        if not hasattr(args, 'seq_len'): args.seq_len = 2**13
        if not hasattr(args, 'level'): args.level = 8
        if not hasattr(args, 'archi'): args.archi = 'DWT'
        if not hasattr(args, 'wavelet'): args.wavelet = 'db4'
        if not hasattr(args, 'num_channels'): args.num_channels = 7
        if not hasattr(args, 'learning_rate'): args.learning_rate = 0.01
        if not hasattr(args, 'batch_size'): args.batch_size = 8
        if not hasattr(args, 'train_epochs'): args.train_epochs = 1000
        if not hasattr(args, 'lambda_l1'): args.lambda_l1 = 1.0
        if not hasattr(args, 'patience'): args.patience = 10
        if not hasattr(args, 'use_amp'): args.use_amp = False
        if not hasattr(args, 'checkpoints'): args.checkpoints = './checkpoints/'
        if not hasattr(args, 'save_model'): args.save_model = True
        
        # Data loader parameters
        if not hasattr(args, 'augmentation_ratio'): args.augmentation_ratio = 0
        if not hasattr(args, 'embed'): args.embed = 'timeF'
        if not hasattr(args, 'label_len'): args.label_len = 48
        if not hasattr(args, 'pred_len'): args.pred_len = 96
        if not hasattr(args, 'seasonal_patterns'): args.seasonal_patterns = 'Monthly'
        if not hasattr(args, 'num_workers'): args.num_workers = 10
        
        self.device = self._acquire_device()
        
        # Initialize independent models (one for each channel)
        print(f"Initializing {self.args.num_channels} independent models for parallel training...")
        self.models_list = nn.ModuleList()
        for ch in range(self.args.num_channels):
            model = self._build_single_model().to(self.device)
            self.models_list.append(model) 
        
        
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')
      
    def _acquire_device(self):
        if hasattr(self.args, 'use_gpu') and self.args.use_gpu:
            if hasattr(self.args, 'gpu_type') and self.args.gpu_type == 'cuda':
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not hasattr(self.args, 'use_multi_gpu') or not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            elif hasattr(self.args, 'gpu_type') and self.args.gpu_type == 'mps':
                device = torch.device('mps')
                print('Use GPU: mps')
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f'Use GPU: {device}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_single_model(self):
        model = NormalizedWaveletAutoencoder(
            seq_len=self.args.seq_len,
            level=self.args.level,
            archi=self.args.archi,
            wavelet=self.args.wavelet
        )
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    

    def forward_parallel(self, x, criterion_fn=None, lambda_val=None):
        """
        Run ensemble models for each channel using simple loop.
        Input x shape: [num_channels, batch, 1, seq_len]
        """
        # Collect results for each channel
        results_list = []
        
        for ch_idx in range(self.args.num_channels):
            # Get channel-specific data
            x_ch = x[ch_idx]  # [batch, 1, seq_len]
            
            # Get the model for this channel
            model = self.models_list[ch_idx]
            
            # Compute loss for this channel
            result = model.compute_loss(x_ch, criterion_fn, lambda_val)
            results_list.append(result)
        
        # Stack results: each result is a tuple (loss, recon_loss, sparse_loss)
        # Convert list of tuples to tuple of tensors
        total_losses = torch.stack([r[0] for r in results_list])
        recon_losses = torch.stack([r[1] for r in results_list])
        sparse_losses = torch.stack([r[2] for r in results_list])
        
        return total_losses, recon_losses, sparse_losses

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def compare_wavelets(self, ch_idx):
        """
        Compare different wavelet initializations (db2, db3, db4) against the trained model
        for a specific channel. This runs serially.
        """
        criterion = self._select_criterion()
        wavelets = ['db2', 'db3', 'db4']
        trained_wavelet = self.args.wavelet
        
        print(f"\n    Comparing wavelets for Channel {ch_idx + 1} on test set:")
        print(f"    Trained wavelet: {trained_wavelet}")
        
        results = {}
        
        # 1. Test untrained baselines
        for wavelet_name in wavelets:
            # Create fresh model instance
            test_model = NormalizedWaveletAutoencoder(
                seq_len=self.args.seq_len,
                level=self.args.level,
                archi=self.args.archi,
                wavelet=wavelet_name
            ).to(self.device)
            test_model.eval()
            
            total_loss = 0.0
            batch_count = 0
            
            with torch.no_grad():
                # Loop through test data serially
                iterator = self.test_loader 
                for i, item in enumerate(iterator):
                    if i >= 20: break 
                    
                    batch_x = item[0].float().to(self.device)
                        
                    # Extract single channel [batch, 1, seq_len]
                    if batch_x.shape[-1] > ch_idx:
                        x_ch = batch_x[:, :, ch_idx].unsqueeze(1)
                    else:
                        continue
                        
                    loss, _, _ = test_model.compute_loss(x_ch, criterion, self.args.lambda_l1)
                    total_loss += loss.item()
                    batch_count += 1
                    
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            results[wavelet_name] = avg_loss

        # 2. Test the Trained Model
        # Use the trained model directly from models_list
        trained_model = self.models_list[ch_idx]
        trained_model.eval()
        
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            iterator = self.test_loader 
            for i, item in enumerate(iterator):
                if i >= 20: break
                
                batch_x = item[0].float().to(self.device)
                    
                if batch_x.shape[-1] > ch_idx:
                    x_ch = batch_x[:, :, ch_idx].unsqueeze(1)
                else:
                    continue
                    
                loss, _, _ = trained_model.compute_loss(x_ch, criterion, self.args.lambda_l1)
                total_loss += loss.item()
                batch_count += 1
                
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        trained_label = f'Trained ({trained_wavelet})'
        results[trained_label] = avg_loss

        # Print results table
        print(f"      {'Model':<25s} | {'Loss':>10s} | {'Reduction':>12s}")
        print(f"      " + "-" * 55)
        
        for wavelet_name in wavelets:
            print(f"      {wavelet_name + ' (untrained)':<25s} | {results[wavelet_name]:>10.6f} | {'-':>12s}")
            
        print(f"      " + "-" * 55)
        untrained_loss = results.get(trained_wavelet, results.get('db4', 0))
        trained_loss = results[trained_label]
        reduction = untrained_loss - trained_loss
        reduction_percent = (reduction / untrained_loss * 100) if untrained_loss > 0 else 0
        
        print(f"      {trained_label:<25s} | {trained_loss:>10.6f} | {reduction:>10.6f} ({reduction_percent:>5.2f}%)")

    def train(self, setting):
        print(f"Self-supervised Training on {self.args.num_channels} channels")
        print(f"Epochs: {self.args.train_epochs}, Batch size: {self.args.batch_size}")
        print("=" * 80)
        
        path = os.path.join(self.args.checkpoints, 'autoencoding', self.args.data)
        if self.args.save_model and not os.path.exists(path):
            os.makedirs(path)
            
        criterion = self._select_criterion()
        # Optimizer for all models' parameters
        optimizer = optim.Adam(self.models_list.parameters(), lr=self.args.learning_rate)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.train_epochs):
            # Training Phase
            train_loss_sum = 0.0
            batch_count = 0
            
            iterator = self.train_loader 
            
            for i, item in enumerate(iterator):
                batch_x = item[0].float().to(self.device)
                
                # Key Data Transformation:
                # Original Input: [batch, seq_len, channels]
                # Target Input for vmap: [channels, batch, 1, seq_len]
                x_parallel = batch_x.permute(2, 0, 1) # -> [channels, batch, seq_len]
                x_parallel = x_parallel.unsqueeze(2)  # -> [channels, batch, 1, seq_len]
                
                optimizer.zero_grad()
                
                # Parallel Forward Pass
                # total_loss_ch shape: [num_channels]
                total_loss_ch, _, _ = self.forward_parallel(x_parallel, criterion, self.args.lambda_l1)
                
                # Sum losses across all channels to update all models simultaneously
                loss = total_loss_ch.sum()
                
                loss.backward()
                optimizer.step()
                
                # Record average loss per channel for reporting
                train_loss_sum += loss.item() / self.args.num_channels
                batch_count += 1
                
            avg_train_loss = train_loss_sum / batch_count
            
            # Validation Phase (Parallelized)
            val_loss_sum = 0.0
            val_count = 0
            
            iterator_val = self.vali_loader 
            with torch.no_grad():
                for i, item in enumerate(iterator_val):
                    batch_x = item[0].float().to(self.device)
                        
                    x_parallel = batch_x.permute(2, 0, 1).unsqueeze(2)
                    total_loss_ch, _, _ = self.forward_parallel(x_parallel, criterion, self.args.lambda_l1)
                    val_loss_sum += total_loss_ch.sum().item() / self.args.num_channels
                    val_count += 1
            
            avg_val_loss = val_loss_sum / val_count
            
            print(f"Epoch {epoch + 1}/{self.args.train_epochs} - Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if self.args.save_model:
                    # Save all channel models
                    torch.save(self.models_list.state_dict(), os.path.join(path, 'parallel_checkpoint.pth'))
                    print("  Best model saved.")

        return self.models_list

    def test(self, setting=None):
        print("\n" + "=" * 80)
        print("Testing (Parallel)...")
        criterion = self._select_criterion()
        
        # Load best model if available
        path = os.path.join(self.args.checkpoints, 'autoencoding', self.args.data)
        best_path = os.path.join(path, 'parallel_checkpoint.pth')
        if self.args.save_model and os.path.exists(best_path):
            self.models_list.load_state_dict(torch.load(best_path))
            print(f"Loaded best model from {best_path}")

        total_loss = 0.0
        count = 0
        
        iterator = self.test_loader 
        with torch.no_grad():
            for i, item in enumerate(iterator):
                batch_x = item[0].float().to(self.device)
                
                x_parallel = batch_x.permute(2, 0, 1).unsqueeze(2)
                total_loss_ch, _, _ = self.forward_parallel(x_parallel, criterion, self.args.lambda_l1)
                
                total_loss += total_loss_ch.mean().item()
                count += 1
        
        print(f"Test - Overall Average Loss: {total_loss / count:.6f}")
        
        # Perform serial comparison for each channel to verify improvement per channel
        # This uses the 'compare_wavelets' function restored as requested
        print("\n" + "=" * 80)
        print("Detailed Comparison (Serial)...")
        for ch in range(self.args.num_channels):
            self.compare_wavelets(ch)
            
        return total_loss / count