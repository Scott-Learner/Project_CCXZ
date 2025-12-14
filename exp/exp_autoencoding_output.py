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
import pandas as pd
from layers.NeuralDWAV import NeuralDWAV

warnings.filterwarnings('ignore')
print("LOADED exp_autoencoding_output FROM:", __file__)


class ElectricityWindowGenerator:
    """
    Random sliding-window generator from electricity.csv for columns ["5", "15"].

    Returns:
        batch_x, batch_y: torch.FloatTensor of shape [batch, seq_len, num_channels]
    """

    def __init__(self, csv_path: str, seq_len: int, columns=("5", "15"), normalize=True):
        self.csv_path = csv_path
        self.seq_len = int(seq_len)
        self.columns = tuple(columns)
        self.normalize = bool(normalize)

        df = pd.read_csv(self.csv_path)

        print("CSV PATH:", self.csv_path)
        print("Requested columns:", self.columns, "len=", len(self.columns))
        print("Selected df shape:", df.loc[:, list(self.columns)].shape)


        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}. Available columns: {list(df.columns)[:30]} ...")

        data = df.loc[:, list(self.columns)].to_numpy(dtype=np.float32)  # [T, C]

        if self.normalize:
            self.mean = data.mean(axis=0, keepdims=True)
            self.std = data.std(axis=0, keepdims=True) + 1e-8
            data = (data - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        self.data = data
        self.num_channels = data.shape[1]
        print("num_channel = " + str(data.shape[1]))
        self.T = data.shape[0]
        self.num_windows = self.T - self.seq_len + 1

        if self.num_windows <= 0:
            raise ValueError(
                f"seq_len={self.seq_len} is too long for electricity.csv length={self.T}. "
                f"Reduce args.seq_len."
            )

    def __getitem__(self, batch_size: int):
        batch_size = int(batch_size)
        idx = np.random.randint(0, self.num_windows, size=batch_size)
        batch = np.stack([self.data[i:i + self.seq_len, :] for i in idx], axis=0)
        return torch.from_numpy(batch).float(), torch.from_numpy(batch).float()


class Exp_Autoencoding(Exp_Basic):
    """Experiment class for autoencoding with NeuralDWAV - per-channel training"""

    def __init__(self, args):
        self.args = args

        if not hasattr(args, 'seq_len'):
            args.seq_len = 2**13
        if not hasattr(args, 'level'):
            args.level = 8
        if not hasattr(args, 'archi'):
            args.archi = 'DWT'
        if not hasattr(args, 'wavelet'):
            args.wavelet = 'db4'
        if not hasattr(args, 'learning_rate'):
            args.learning_rate = 0.01
        if not hasattr(args, 'batch_size'):
            args.batch_size = 8
        if not hasattr(args, 'train_epochs'):
            args.train_epochs = 1000
        if not hasattr(args, 'lambda_l1'):
            args.lambda_l1 = 1.0
        if not hasattr(args, 'log_interval'):
            args.log_interval = 100
        if not hasattr(args, 'patience'):
            args.patience = 10
        if not hasattr(args, 'use_amp'):
            args.use_amp = False

        # optional config
        if not hasattr(args, 'electricity_cols'):
            args.electricity_cols = ("5", "15")
        if not hasattr(args, 'normalize_electricity'):
            args.normalize_electricity = True

        self.device = self._acquire_device()

        # IMPORTANT: electricity.csv is in Project_CCXZ/exp/ per your screenshot
        csv_path = os.path.join(os.path.dirname(__file__), "electricity.csv")
        self.generator = ElectricityWindowGenerator(
            csv_path=csv_path,
            seq_len=self.args.seq_len,
            columns=self.args.electricity_cols,
            normalize=self.args.normalize_electricity
        )

        self.args.num_channels = self.generator.num_channels

        self.models = []
        for _ in range(self.args.num_channels):
            model = self._build_single_model().to(self.device)
            self.models.append(model)

        self.log_signal, _ = self.generator.__getitem__(1)
        self.log_signal = self.log_signal.to(self.device)

    def _get_results_base(self, setting: str) -> str:
        """
        Save outputs into Project_CCXZ/results/autoencoding/setting/...

        This file is inside Project_CCXZ/exp/, so project_root = parent(exp_dir).
        """
        exp_dir = os.path.dirname(__file__)                 # .../Project_CCXZ/exp
        project_root = os.path.dirname(exp_dir)             # .../Project_CCXZ
        out_base = os.path.join(project_root, "results", "autoencoding", setting)
        os.makedirs(out_base, exist_ok=True)
        return out_base

    def _acquire_device(self):
        if hasattr(self.args, 'use_gpu') and self.args.use_gpu:
            if hasattr(self.args, 'gpu_type') and self.args.gpu_type == 'cuda':
                return torch.device('cuda')
            elif hasattr(self.args, 'gpu_type') and self.args.gpu_type == 'mps':
                return torch.device('mps')
        return torch.device('cpu')

    def _build_single_model(self):
        return NeuralDWAV(
            Input_Size=self.args.seq_len,
            Input_Level=self.args.level,
            Input_Archi=self.args.archi,
            Filt_Trans=True,
            Filt_Train=True,
            Filt_Tfree=False,
            Filt_Style="Filter_Free",
            Filt_Mother=self.args.wavelet,
            Act_Train=True,
            Act_Style="Sigmoid",
            Act_Symmetric=True,
            Act_Init=0
        )

    def _select_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), eps=1e-7)

    def _select_criterion(self):
        return nn.L1Loss()

    def train(self, setting):
        print(f"Training {self.args.num_channels} channels independently")
        print(f"Epochs: {self.args.train_epochs}, Batch size: {self.args.batch_size}, Lambda: {self.args.lambda_l1}")
        print("=" * 80)

        # ✅ save everything under Project_CCXZ/results/autoencoding/setting/
        path = self._get_results_base(setting)
        print("✅ Outputs will be saved to:", os.path.abspath(path))

        criterion = self._select_criterion()
        log_interval = getattr(self.args, 'log_interval', 100) #################################

        for ch_idx in range(self.args.num_channels):
            print(f"\n🔹 Training Channel {ch_idx + 1}/{self.args.num_channels}")

            model = self.models[ch_idx]
            model_optim = self._select_optimizer(model)
            model.train()

            start_time = time.time()

            self._log_wavelets_and_sweeps(model, ch_idx, epoch=0, setting=setting, base_path=path)

            for epoch in range(self.args.train_epochs):
                batch_x, batch_y = self.generator.__getitem__(self.args.batch_size)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                x_ch = batch_x[:, :, ch_idx].unsqueeze(1)

                model.zero_grad()
                Emb = model.T(x_ch)
                loss_recon = criterion(model.iT(Emb), x_ch)
                loss_sparse = model.L1_sum(Emb)
                loss = loss_recon + self.args.lambda_l1 * loss_sparse

                loss.backward()
                model_optim.step()

                if log_interval is not None and log_interval > 0 and (epoch + 1) % log_interval == 0:
                    self._log_wavelets_and_sweeps(model, ch_idx, epoch=epoch + 1, setting=setting, base_path=path)

            model.eval()
            elapsed_time = time.time() - start_time

            # Save model checkpoint into same results folder
            model_path = os.path.join(path, f'channel_{ch_idx}_checkpoint.pth')
            torch.save(model.state_dict(), model_path)

            print(f"   ✓ Channel {ch_idx + 1} saved checkpoint → {model_path} | Time: {elapsed_time:.2f}s")

        print("\n" + "=" * 80)
        print("✅ All channels trained successfully!")
        return self.models

    def _log_wavelet_shapes_at_epoch(self, model, ch_idx, epoch, setting, base_path):
        shapes_dir = os.path.join(base_path, 'wavelet_shapes')
        os.makedirs(shapes_dir, exist_ok=True)

        Filt = model.Filt
        kernel_list = Filt.kernel
        positions = Filt.position

        rows = []
        for lvl in range(1, Filt.level + 1):
            pos_list = positions[lvl]
            for kpos, k_idx in enumerate(pos_list):
                ker = kernel_list[k_idx].detach().cpu().numpy().reshape(-1)
                row = {
                    'epoch': epoch,
                    'channel': ch_idx,
                    'level': lvl,
                    'kernel_pos': kpos,
                    'kernel_index': int(k_idx),
                }
                for n, c in enumerate(ker):
                    row[f'c_{n}'] = float(c)
                rows.append(row)

        if not rows:
            return

        df = pd.DataFrame(rows)
        csv_path = os.path.join(shapes_dir, f'wavelet_shapes_channel{ch_idx}.csv')
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode='a', index=False, header=header)

    def _log_wavelet_sweeps_at_epoch(self, model, ch_idx, epoch, setting, base_path):
        sweeps_dir = os.path.join(base_path, 'wavelet_sweeps_over_time')
        os.makedirs(sweeps_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():
            if not hasattr(self, 'log_signal') or self.log_signal is None:
                batch_x, _ = self.generator.__getitem__(1)
                self.log_signal = batch_x.to(self.device)

            x_all = self.log_signal.to(self.device)
            x = x_all[:, :, ch_idx].unsqueeze(1)

            Emb = model.T(x)
            L = len(Emb) - 1

            recons = []
            for delete_until in range(L, -1, -1):
                Emb_k = []
                for j in range(len(Emb)):
                    if j >= delete_until:
                        Emb_k.append(Emb[j])
                    else:
                        Emb_k.append(torch.zeros_like(Emb[j]))

                x_k = model.iT(Emb_k)
                recons.append(x_k.squeeze().cpu().numpy())

        arr = np.stack(recons, axis=1)
        columns = [f'x_{i}' for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=columns)
        df['original'] = x.squeeze().cpu().numpy()
        csv_path = os.path.join(
            sweeps_dir,
            f'ch{ch_idx}_ep{epoch:04d}_sweeps.csv'
        )

        df.to_csv(csv_path, index=False)

    def _log_wavelets_and_sweeps(self, model, ch_idx, epoch, setting, base_path):
        log_interval = getattr(self.args, 'log_interval', 10)
        if log_interval is None or log_interval <= 0:
            return
        self._log_wavelet_shapes_at_epoch(model, ch_idx, epoch, setting, base_path)
        self._log_wavelet_sweeps_at_epoch(model, ch_idx, epoch, setting, base_path)

    def export_wavelet_sweeps(self, setting, outdir=None):
        # ✅ default to Project_CCXZ/results/autoencoding/<setting>/wavelet_sweeps/
        if outdir is None:
            base = self._get_results_base(setting)
            outdir = os.path.join(base, "wavelet_sweeps")
        os.makedirs(outdir, exist_ok=True)

        ch_idx = 0
        model = self.models[ch_idx]
        model.eval()

        batch_x, _ = self.generator.__getitem__(1)
        batch_x = batch_x.to(self.device)

        x = batch_x[:, :, ch_idx].unsqueeze(1)

        Emb = model.T(x)
        L = len(Emb) - 1

        recons = []
        for delete_until in range(L, -1, -1):
            Emb_k = []
            for j in range(len(Emb)):
                if j >= delete_until:
                    Emb_k.append(Emb[j])
                else:
                    Emb_k.append(torch.zeros_like(Emb[j]))

            x_k = model.iT(Emb_k)
            recons.append(x_k.squeeze().cpu().numpy())

        arr = np.stack(recons, axis=1)
        columns = [f'x_{i}' for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=columns)
        df['original'] = x.squeeze().cpu().numpy()

        csv_path = os.path.join(outdir, f'{setting}_wavelet_sweeps.csv')
        df.to_csv(csv_path, index=False)

        print(f"Saved sweep CSV → {csv_path}")
