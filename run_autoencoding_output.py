import argparse
import os
import torch
import random
import numpy as np
from exp.exp_autoencoding_output import Exp_Autoencoding

if __name__ == '__main__':
    # Fix random seed for reproducibility
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='NeuralDWAV Autoencoding')

    # Basic config
    parser.add_argument('--task_name', type=str, default='autoencoding',
                        help='task name')
    parser.add_argument('--is_training', type=int, default=1,
                        help='status: 1=training, 0=testing')
    parser.add_argument('--model_id', type=str, default='autoencoding_test',
                        help='model id')
    parser.add_argument('--model', type=str, default='NeuralDWAV',
                        help='model name')

    # Data config
    parser.add_argument('--data', type=str, default='ETTh1',
                        help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv',
                        help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='M=multivariate, S=univariate, MS=multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features: s,t,h,d,b,w,m')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')

    # Autoencoding parameters
    parser.add_argument('--seq_len', type=int, default=8192,
                        help='signal sequence length')
    parser.add_argument('--level', type=int, default=3,
                        help='wavelet decomposition level')
    parser.add_argument('--archi', type=str, default='DWT',
                        help='architecture: DWT or WPT')
    parser.add_argument('--wavelet', type=str, default='db4',
                        help='wavelet type: db2, db3, db4, etc.')
    parser.add_argument('--num_channels', type=int, default=7,
                        help='number of channels')

    # Training parameters (following LDWT_main DESPAWN)
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='optimizer learning rate')
    parser.add_argument('--lambda_l1', type=float, default=1.0,
                        help='L1 sparsity regularization weight')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size (small batch works better)')
    parser.add_argument('--train_epochs', type=int, default=1000,
                        help='train epochs per channel')
    parser.add_argument('--patience', type=int, default=20,
                        help='early stopping patience')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='use gpu')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu device id')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='device ids of multiple gpus')

    # Experiment settings
    parser.add_argument('--itr', type=int, default=1,
                        help='experiments times')
    parser.add_argument('--des', type=str, default='autoencoding',
                        help='exp description')

    # Data loader settings
    parser.add_argument('--use_real_data', action='store_true',
                        help='use real dataset instead of dummy generator', default=False)
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding')
    parser.add_argument('--label_len', type=int, default=48,
                        help='for data loader compatibility')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='for data loader compatibility')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                        help='for M4 dataset')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='data loader num workers')
    parser.add_argument('--augmentation_ratio', type=int, default=0,
                        help='data augmentation ratio')

    # Output settings
    parser.add_argument('--save_wavelet_sweeps', action='store_true',
                        help='export x0..x_L reconstructions to CSV', default=False)
    parser.add_argument('--sweeps_outdir', type=str, default='./wavelet_sweeps/',
                        help='directory to store CSV files for x0..x_L')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Set device based on args
    if args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')

    # Run experiments
    for ii in range(args.itr):
        # Setting record of experiments
        setting = '{}_{}_{}_{}_{}_L{}_ch{}_lr{}_lambda{}_bs{}_ep{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.wavelet,
            args.level,
            args.num_channels,
            args.learning_rate,
            args.lambda_l1,
            args.batch_size,
            args.train_epochs,
            ii
        )

        print('\n' + '=' * 80)
        print(f'Iteration {ii + 1}/{args.itr}')
        print('=' * 80)

        # Initialize experiment
        exp = Exp_Autoencoding(args)

        if args.is_training:
            exp.train(setting)
            exp.vali(setting)
            exp.test(setting)
        else:
            exp.test(setting)

        # NEW: export x0,x1,...,x_L (e.g. x0..x3) to a CSV
        if args.save_wavelet_sweeps:
            # You implement this method inside Exp_Autoencoding:
            #   def export_wavelet_sweeps(self, setting, outdir): ...
            exp.export_wavelet_sweeps(setting, outdir=args.sweeps_outdir)

        torch.cuda.empty_cache()

    print('\n' + '=' * 80)
    print('All experiments completed!')
    print('=' * 80)
