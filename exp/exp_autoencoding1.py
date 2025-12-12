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

class Generator_dummy_test():  # Stepwise function with gaussian noise
    """Synthetic data generator for autoencoding task - multi-channel version"""
    def __init__(self, Np=2**13, num_channels=7):
        np.random.seed(42)
        self.Nchange = 5
        self.Np = Np
        self.sigma = 0.2
        self.num_channels = num_channels
        
    def __getitem__(self, batchIndex):
        batchX, batchY = self.generate_batch(batchIndex)
        # For autoencoding: [batch, seq_len, channel] format like real data
        # X: noisy, Y: clean
        return (torch.tensor(batchX).float(), 
                torch.tensor(batchY).float())

    def generate_batch(self, batchIndex):
        # Generate [batch, seq_len, channel] shaped data
        batchX = np.zeros((batchIndex, self.Np, self.num_channels))
        batchY = np.zeros((batchIndex, self.Np, self.num_channels))
        
        for batch in range(batchIndex):
            for ch in range(self.num_channels):
                # Each channel has independent stepwise signal
                x = np.random.randn(self.Nchange)
                y = np.concatenate(([0], np.floor(np.sort(np.random.rand(self.Nchange)) * self.Np).astype(int)))
                for i in range(self.Nchange):
                    batchY[batch, y[i]:y[i+1], ch] = x[i]
                batchX[batch, :, ch] = batchY[batch, :, ch] + self.sigma * np.random.randn(self.Np)
        
        return batchX, batchY

class Exp_Autoencoding(Exp_Basic):
    """Experiment class for autoencoding with NeuralDWAV - per-channel training"""
    
    def __init__(self, args):
        # Set default values for autoencoding task
        self.args = args
        
        # For autoencoding, we use NeuralDWAV directly instead of models from model_dict
        # Set default parameters if not provided
        if not hasattr(args, 'seq_len'):
            args.seq_len = 2**13  # Default signal length
        if not hasattr(args, 'level'):
            args.level = 8
        if not hasattr(args, 'archi'):
            args.archi = 'DWT'
        if not hasattr(args, 'num_channels'):
            args.num_channels = 7  # Default: 7 channels like ETT data
        if not hasattr(args, 'learning_rate'):
            args.learning_rate = 0.01  # Same as DESPAWN in LDWT_main
        if not hasattr(args, 'batch_size'):
            args.batch_size = 8
        if not hasattr(args, 'train_epochs'):
            args.train_epochs = 1000  # Same as LDWT_main
        if not hasattr(args, 'lambda_l1'):
            args.lambda_l1 = 1.0  # L1 regularization weight
        if not hasattr(args, 'patience'):
            args.patience = 10
        if not hasattr(args, 'use_amp'):
            args.use_amp = False
        if not hasattr(args, 'checkpoints'):
            args.checkpoints = './checkpoints/'
        
        # Don't call parent init, we'll handle device setup manually
        self.device = self._acquire_device()
        
        # Create separate NeuralDWAV model for each channel
        self.models = []
        for ch in range(self.args.num_channels):
            model = self._build_single_model().to(self.device)
            self.models.append(model)
        
        # Create dummy data generator
        self.generator = Generator_dummy_test(Np=self.args.seq_len, num_channels=self.args.num_channels)

    def _acquire_device(self):
        """Acquire computing device"""
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
        """Build a single NeuralDWAV model for one channel"""
        model = NeuralDWAV(
            Input_Size=self.args.seq_len,
            Input_Level=self.args.level,
            Input_Archi=self.args.archi,
            Filt_Trans=True,
            Filt_Train=True,
            Filt_Tfree=False,
            Filt_Style="Kernel_Free",  # Time-invariant
            Filt_Mother="db4",
            Act_Train=True,
            Act_Style="Sigmoid",
            Act_Symmetric=True,
            Act_Init=0
        ).float()
        
        return model
    
    def _build_model(self):
        """For compatibility with parent class - not used in per-channel training"""
        return self.models[0] if self.models else None
    
    def _get_data(self, flag):
        """Get data from real dataloader or dummy generator"""
        # For autoencoding, we can use either real data or dummy generator
        # If using real data, make sure batch_y = batch_x
        try:
            data_set, data_loader = data_provider(self.args, flag)
            return data_set, data_loader
        except:
            # If data provider fails, use dummy generator
            print(f"Using dummy generator for {flag} data")
            return None, None
    
    def _select_optimizer(self, model):
        """Create optimizer for a single model"""
        model_optim = optim.Adam(model.parameters(), 
                                lr=self.args.learning_rate,
                                betas=(0.9, 0.999), 
                                eps=1e-7)
        return model_optim
    
    def _select_criterion(self):
        # Use L1 loss like DESPAWN in LDWT_main
        criterion = nn.L1Loss()
        return criterion
    
    def compare_wavelets(self, ch_idx, batch_x, batch_y, trained_model=None):
        """
        Compare different wavelet initializations (db2, db3, db4) with same parameters
        
        Args:
            ch_idx: channel index
            batch_x: input data [batch, seq_len, channels]
            batch_y: target data [batch, seq_len, channels]
            trained_model: the trained model (optional, for comparison)
        """
        criterion = self._select_criterion()
        wavelets = ['db2', 'db3', 'db4']
        
        print(f"\n   📊 Comparing wavelets for Channel {ch_idx + 1}:")
        
        # Extract single channel
        x_ch = batch_x[:, :, ch_idx].unsqueeze(1)
        y_ch = batch_y[:, :, ch_idx].unsqueeze(1)
        
        results = {}
        
        # Test each wavelet
        for wavelet_name in wavelets:
            # Create new model with different wavelet initialization
            test_model = NeuralDWAV(
                Input_Size=self.args.seq_len,
                Input_Level=self.args.level,
                Input_Archi=self.args.archi,
                Filt_Trans=True,
                Filt_Train=True,
                Filt_Tfree=False,
                Filt_Style="Kernel_Free",
                Filt_Mother=wavelet_name,  # Different wavelet
                Act_Train=True,
                Act_Style="Sigmoid",
                Act_Symmetric=True,
                Act_Init=0
            ).float().to(self.device)
            
            test_model.eval()
            
            with torch.no_grad():
                # Compute loss with untrained model (just initialization)
                Emb = test_model.T(x_ch)
                loss_recon = criterion(test_model.iT(Emb), x_ch)
                loss_sparse = test_model.L1_sum(Emb)
                loss = loss_recon + self.args.lambda_l1 * loss_sparse
                
                results[wavelet_name] = loss.item()
        
        # If trained model provided, also show its loss
        if trained_model is not None:
            trained_model.eval()
            with torch.no_grad():
                Emb = trained_model.T(x_ch)
                loss_recon = criterion(trained_model.iT(Emb), x_ch)
                loss_sparse = trained_model.L1_sum(Emb)
                loss = loss_recon + self.args.lambda_l1 * loss_sparse
                results['Trained (db4)'] = loss.item()
        
        # Print comparison
        print(f"      Wavelet Init | Loss (Untrained)")
        print(f"      " + "-" * 35)
        for wavelet_name in wavelets:
            print(f"      {wavelet_name:12s} | {results[wavelet_name]:.6f}")
        
        if trained_model is not None:
            print(f"      " + "-" * 35)
            print(f"      {'Trained (db4)':12s} | {results['Trained (db4)']:.6f}")
  
        
        return results
    
    def vali(self, setting=None):
        """Validation loop for all channels"""
        criterion = self._select_criterion()
        channel_losses = []
        
        for ch_idx in range(self.args.num_channels):
            model = self.models[ch_idx]
            model.eval()
            
            total_loss = []
            with torch.no_grad():
                num_val_batches = 10
                for i in range(num_val_batches):
                    batch_x, batch_y = self.generator.__getitem__(self.args.batch_size)
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Extract single channel
                    x_ch = batch_x[:, :, ch_idx].unsqueeze(1)
                    
                    # Dual L1 loss: reconstruction + L1 sparsity
                    Emb = model.T(x_ch)
                    loss_recon = criterion(model.iT(Emb), x_ch)
                    loss_sparse = model.L1_sum(Emb)
                    loss = loss_recon + self.args.lambda_l1 * loss_sparse
                    
                    total_loss.append(loss.item())
            
            avg_loss = np.average(total_loss)
            channel_losses.append(avg_loss)
            model.train()
        
        overall_avg = np.average(channel_losses)
        print(f"Validation - Overall Loss: {overall_avg:.6f}")
        
        return overall_avg
    
    def train(self, setting):
        """Training loop - per-channel independent training like LDWT_main"""
        print(f"Training {self.args.num_channels} channels independently")
        print(f"Epochs: {self.args.train_epochs}, Batch size: {self.args.batch_size}, Lambda: {self.args.lambda_l1}")
        print("=" * 80)
        
        # Create checkpoint directory
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        criterion = self._select_criterion()
        
        # Train each channel independently
        for ch_idx in range(self.args.num_channels):
            print(f"\n🔹 Training Channel {ch_idx + 1}/{self.args.num_channels}")
            
            model = self.models[ch_idx]
            model_optim = self._select_optimizer(model)
            model.train()
            
            # Training loop (following LDWT_main style)
            start_time = time.time()
            
            for epoch in range(self.args.train_epochs):
                # Get batch from generator
                batch_x, batch_y = self.generator.__getitem__(self.args.batch_size)
                batch_x = batch_x.to(self.device)  # [batch, seq_len, channels]
                batch_y = batch_y.to(self.device)
                
                # Extract single channel: [batch, seq_len] -> [batch, 1, seq_len]
                x_ch = batch_x[:, :, ch_idx].unsqueeze(1)
                y_ch = batch_y[:, :, ch_idx].unsqueeze(1)
                
                model.zero_grad()
                
                # Forward pass with dual L1 loss (like DESPAWN in LDWT_main)
                Emb = model.T(x_ch)  # Get wavelet coefficients
                
                # Loss = reconstruction loss + lambda * L1 sparsity
                loss_recon = criterion(model.iT(Emb), x_ch)
                loss_sparse = model.L1_sum(Emb)
                loss = loss_recon + self.args.lambda_l1 * loss_sparse
                
                loss.backward()
                model_optim.step()
                
            # After training, evaluate the channel
            model.eval()
            elapsed_time = time.time() - start_time
            
            # Final evaluation
            final_loss = 0.0
            num_test_batches = 10
            with torch.no_grad():
                for _ in range(num_test_batches):
                    batch_x, batch_y = self.generator.__getitem__(self.args.batch_size)
                    batch_x = batch_x.to(self.device)
                    x_ch = batch_x[:, :, ch_idx].unsqueeze(1)
                    
                    Emb = model.T(x_ch)
                    loss_recon = criterion(model.iT(Emb), x_ch)
                    loss_sparse = model.L1_sum(Emb)
                    loss = loss_recon + self.args.lambda_l1 * loss_sparse
                    final_loss += loss.item()
            
            final_loss /= num_test_batches
            
            print(f"   ✓ Channel {ch_idx + 1} - Final Loss: {final_loss:.6f} | Time: {elapsed_time:.2f}s")
            
            # Compare with different wavelets
            # Get a fresh test batch for comparison
            test_batch_x, test_batch_y = self.generator.__getitem__(self.args.batch_size)
            test_batch_x = test_batch_x.to(self.device)
            test_batch_y = test_batch_y.to(self.device)
            self.compare_wavelets(ch_idx, test_batch_x, test_batch_y, trained_model=model)
            
            # Save model
            model_path = os.path.join(path, f'channel_{ch_idx}_checkpoint.pth')
            torch.save(model.state_dict(), model_path)
        
        print("\n" + "=" * 80)
        print("✅ All channels trained successfully!")
        
        return self.models
    
    def test(self, setting=None, test_loader=None):
        """Test loop - evaluate all channels"""
        criterion = self._select_criterion()
        channel_losses = []
        
        for ch_idx in range(self.args.num_channels):
            model = self.models[ch_idx]
            model.eval()
            
            total_loss = []
            with torch.no_grad():
                num_test_batches = 20
                for i in range(num_test_batches):
                    batch_x, batch_y = self.generator.__getitem__(self.args.batch_size)
                    batch_x = batch_x.to(self.device)
                    
                    # Extract single channel
                    x_ch = batch_x[:, :, ch_idx].unsqueeze(1)
                    
                    # Dual L1 loss
                    Emb = model.T(x_ch)
                    loss_recon = criterion(model.iT(Emb), x_ch)
                    loss_sparse = model.L1_sum(Emb)
                    loss = loss_recon + self.args.lambda_l1 * loss_sparse
                    
                    total_loss.append(loss.item())
            
            avg_loss = np.average(total_loss)
            channel_losses.append(avg_loss)
        
        overall_avg = np.average(channel_losses)
        print(f"Test - Overall Loss: {overall_avg:.6f}")
        
        return overall_avg

    def export_wavelet_sweeps(self, setting, outdir='./wavelet_sweeps'):
        import pandas as pd
        os.makedirs(outdir, exist_ok=True)

        # Only do one channel for now – modify if you want per-channel outputs
        ch_idx = 0
        model = self.models[ch_idx]
        model.eval()

        # Get a batch
        batch_x, _ = self.generator.__getitem__(1)
        batch_x = batch_x.to(self.device)

        # Extract one channel, shape [1,1,seq_len]
        x = batch_x[:, :, ch_idx].unsqueeze(1)

        with torch.no_grad():

            # ---- 1. Forward transform (L-DWT) ----
            Emb = model.T(x)  # coefficients dict

            # NeuralDWAV stores coefficients in Emb:
            # Emb['A'] is approximation
            # Emb['D'][0], Emb['D'][1], ... detail levels

            A = Emb['A']
            D_list = Emb['D']
            L = len(D_list)

            recons = []

            # ---- 2. Build x0, x1, ..., xL ----
            for delete_until in range(L+1, -1, -1): # before: range(L+1) 
                D_used = []
                for j in range(L):
                    if j >= delete_until: # Levels [delete_until, ..., L] used in reconstruction
                        D_used.append(D_list[j])
                    else: # Levels [0, ..., delete_until-1] left out in reconstruction
                        D_used.append(torch.zeros_like(D_list[j])) 

                Emb_k = {'A': A, 'D': D_used}

                # ---- 3. Inverse transform ----
                x_k = model.iT(Emb_k)

                recons.append(x_k.squeeze().cpu().numpy())

        # ---- 4. Save CSV ----
        arr = np.stack(recons, axis=1)
        columns = [f'x_{i}' for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=columns)

        csv_path = os.path.join(outdir, f'{setting}_wavelet_sweeps.csv')
        df.to_csv(csv_path, index=False)

        print(f"Saved sweep CSV → {csv_path}")
        
# Utility functions for visualization
def npt(x):
    """Convert torch tensor to numpy array"""
    return x.detach().cpu().numpy().squeeze()


def plt_x(n, X, L):
    """Plot signals"""
    plt.figure(n)
    for i in range(len(X)):
        plt.plot(npt(X[i]))
    plt.legend(L)
    plt.title('Signal Reconstruction')
    

def plt_Emb(n, X, L):
    """Plot embeddings/spectrograms"""
    plt.figure(n, figsize=(8, 12))
    fig, axs = plt.subplots(2)
    for i in range(len(X)):
        axs[i].imshow(20 * np.log10(npt(X[i]) + 1e-3), aspect="auto")
        axs[i].set_title(L[i])
