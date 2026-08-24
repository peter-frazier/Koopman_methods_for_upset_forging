import sys
sys.path.insert(0, "/users/PAS3353/peterfrazier/Simple_Koopman_Forging") # TODO: change to folder containing jax-fem-checkpoint

import argparse
import os
import time
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import savemat
from jax_fem_checkpoint import logger

from model import BLRAN_LD
from train import train as train_blran_ld
from read_dataset import normalize


class WindowDataset(Dataset):
    def __init__(self, X, U, steps):
        self.X = X # (n_traj, T,   n_x)
        self.U = U # (n_traj, T-1, n_u)
        self.steps = steps
        self.n_traj, self.traj_len, _ = X.shape

    def __len__(self):
        return self.n_traj * (self.traj_len - self.steps)

    def __getitem__(self, idx):
        traj_id = idx // (self.traj_len - self.steps)
        step_id = idx % (self.traj_len - self.steps)

        xs = [self.X[traj_id, step_id+p, :] for p in range(self.steps + 1)]
        us = [self.U[traj_id, step_id+p, :] for p in range(self.steps)]
        return xs + us
    

def evaluate_model(model, X, U, sims, return_pred=True, get_errors=True, get_NMSE=True, get_NRMSE=True, device='cpu'):
    '''
    INPUTS
    model: PyTorch model to be evaluated
    X: state dataset
    U: input dataset
    sims: simulated trajectory number
    device: run sims on cpu or gpu

    OUTPUTS
    X_pred: predicted states of X (n_traj, traj_len, n_x)        return_pred=false -> returns empty list
    errors: errors between X and X_pred (n_traj, traj_len, n_x), get_errors=False -> returns empty list
    NMSE: NMSE per-sim (n_traj, traj_len),                       get_NMSE=False -> returns empty list
    NRMSE_sim: running NMRSE per-sim (n_traj, traj_len),         get_NRMSE=False -> returns empty list
    NRMSE: running NMRSE across dataset (traj_len,),             get_NRMSE=False -> returns empty list
    '''
    if get_errors==False and (get_NMSE==True or get_NRMSE==True):
        logger.info('WARNING: cannot calculate NMSE or NRMSE without errors. Skipping calculation.')

    n_traj, traj_len, n_x = X.shape
    X_pred    = np.zeros(shape=X.shape, dtype=np.float32)
    errors    = []
    NMSE      = []
    NRMSE_sim = []
    NRMSE     = []

    with torch.no_grad():
        for i, sim in enumerate(sims):
            start = time.time()

            if device == 'cpu':
                x0 = X[i, :1, :]
                us = U[i, : , :].unsqueeze(0)
            else:
                x0 = X[i, :1, :].cuda()
                us = U[i, : , :].unsqueeze(0).cuda()

            z0      = model.encode(x0)
            z_preds = model.rollout(z0, us)
            x_hat   = torch.cat(
                [x0] + [model.decode(z) for z in z_preds], dim=0
            ).cpu().detach().numpy()

            end = time.time()
            logger.debug(f'Ran bilinear simulation {i+1} of {len(sims)} in {end-start} seconds')

            X_pred[i, :, :] = x_hat.squeeze()

    X = X.cpu().detach().numpy()

    if get_errors:
        errors = X - X_pred

        if not return_pred:
            X_pred = []

        if get_NMSE:
            NMSE = np.linalg.norm(errors, axis=2)/np.linalg.norm(X, axis=2)

        if get_NRMSE:
            NRMSE_sim = np.zeros(shape=(n_traj, traj_len), dtype=np.float32)
            NRMSE     = np.zeros(shape=(traj_len,), dtype=np.float32)
            for i in range(traj_len):
                vnorm = np.linalg.norm(X[:,:i+1,:], axis=2)      #(n_traj, i)
                enorm = np.linalg.norm(errors[:,:i+1,:], axis=2) #(n_traj, i)
                NRMSE_sim[:, i] = (np.sum(enorm**2, axis=1)/np.sum(vnorm**2, axis=1))**0.5 #(n_traj,)
                NRMSE[i]        = (np.sum(enorm**2)/np.sum(vnorm))**0.5 #(,)

    run_stats = {
        'X_pred': X_pred,
        'errors': errors,
        'NMSE': NMSE,
        'NRMSE_sim': NRMSE_sim,
        'NRMSE': NRMSE
    }
    return run_stats


if __name__ == "__main__":

    # -- Arguments -----------------------------------------------------------------

    parser = argparse.ArgumentParser(description='BLRAN-LD -- controlled dynamical systems')

    # Data
    parser.add_argument('--data_name',    default='Isothermal_Plasticity',
                        help='label used for data in folder')
    parser.add_argument('--train_frac', type=float, default=0.7,
                        help='fraction of trajectories used for training')
    parser.add_argument('--test_frac',  type=float, default=0.2,
                        help='fraction of trajectories used for testing')
    parser.add_argument('--shift_frac', type=float, default=0.,
                        help='fraction of trajectories to shift to get different sets')

    # Architecture
    parser.add_argument('--n_z',        type=int,   default=512,
                        help='latent (Koopman) dimension')
    parser.add_argument('--n_h',        type=int,   default=4,
                        help='number of hidden layers')
    parser.add_argument('--activation', type=str,   default='LeakyReLU',
                        help='network width multiplier (hidden layer width = 16*alpha)')
    parser.add_argument('--alpha',      type=int,   default=32,
                        help='network width multiplier (hidden layer width = 16*alpha)')
    parser.add_argument('--init_scale', type=float, default=0.99,
                        help='initial spectral radius of A')

    # Training
    parser.add_argument('--steps',      type=int,   default=10,
                        help='multi-step prediction horizon during training')
    parser.add_argument('--epochs',     type=int,   default=500)
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--wd',         type=float, default=1e-4)
    parser.add_argument('--gradclip',   type=float, default=0.05)
    parser.add_argument('--gamma_id',   type=float, default=1.0)
    parser.add_argument('--gamma_fwd',  type=float, default=1.0)
    parser.add_argument('--gamma_lin',  type=float, default=1.0)
    parser.add_argument('--gamma_eig',  type=float, default=0.0,
                        help='weight on eigenvalue stability loss (0 = disabled)')

    parser.add_argument('--seed',       type=int,   default=0)
    parser.add_argument('--device',     type=str,   default='cpu')
    parser.add_argument('--out_dir',    type=str,   default='metrics',
                        help='directory to save per-run metrics (empty = skip)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

# -- Data ----------------------------------------------------------------------

    # Load necessary info
    logger.debug("Loading data...")
    data_dir  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    data_file = os.path.join(data_dir, args.data_name+'.mat')
    meta_file = os.path.join(data_dir, args.data_name+'.json')

    with open(meta_file, 'r') as f:
        sim_info = json.load(f)
    n_traj   = sim_info['n_traj']
    traj_len = sim_info['traj_len']
    fail_idx = sim_info['fail_idx']
    n_fail   = len(fail_idx)

    with h5py.File(data_file, 'r') as f:
        X = f['X'][:] 
        U = f['U'][:]

    _, _, n_x = X.shape
    _, _, n_u = U.shape

    # Normalize
    logger.debug('Normalizing data on [-1, 1]...')
    X_n, U_n, scale = normalize(X, U)

    # Split training/testing
    logger.debug('Splitting testing/training sets...')
    sims    = np.arange(n_traj)
    sims    = np.delete(sims, fail_idx)
    n_traj -= n_fail
    n_train = int(n_traj*args.train_frac)
    n_test  = int(n_traj*args.test_frac)
    n_valid = n_traj - n_train - n_test
    n_shift = int(n_traj*args.shift_frac)
    sims    = np.roll(sims, shift=n_shift)

    test_sims  = sims[:n_test]
    train_sims = sims[n_test:n_test+n_train]
    valid_sims = sims[n_test+n_train:]
    X_tr, U_tr = X_n[train_sims,:,:], U_n[train_sims,:,:]
    X_va, U_va = X_n[valid_sims,:,:], U_n[valid_sims,:,:]

    # Form PyTorch dataset
    logger.debug('Forming dataset for PyTorch...')
        
    X_tr = torch.from_numpy(X_tr)
    U_tr = torch.from_numpy(U_tr)
    dataset = WindowDataset(X_tr, U_tr, args.steps)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    logger.info(f'Train trajectories: {n_train}  |  Validation trajectories: {n_valid}  |  Test trajectories: {n_test}')
    logger.info(f'Windows per epoch:  {len(dataset)}  |  Batches per epoch: {len(loader)}')

    # -- Model ---------------------------------------------------------------------

    model   = BLRAN_LD(n_x, n_u, args.n_z, args.n_h, args.activation, args.alpha, args.init_scale)
    n_param = sum(p.numel() for p in model.parameters())
    logger.debug(f'BLRAN-LD | n_x={n_x}  n_u={n_u}  n_h={args.n_h}  n_z={args.n_z}  '
                f'alpha={args.alpha} (width={16*args.alpha})  activation={args.activation} | '
                f'B shape=({model.n_z}, {model.n_z * n_u}) | params={n_param:,}')

    # -- Train ---------------------------------------------------------------------

    t0 = time.time()
    history = train_blran_ld(
        model, loader, args.epochs,
        lr=args.lr, wd=args.wd, gradclip=args.gradclip,
        gamma_fwd=args.gamma_fwd, gamma_lin=args.gamma_lin,
        gamma_eig=args.gamma_eig, gamma_id=args.gamma_id,
        device=args.device, print_every=10,
    )
    train_time = time.time() - t0

    metrics_dir = os.path.join(os.path.dirname(__file__), args.out_dir)
    os.makedirs(metrics_dir, exist_ok=True)
    model_path = os.path.join(metrics_dir, 'model.pt')
    torch.save({'state_dict': model.state_dict(), 'args': vars(args),
                'scale': scale, 'n_x': n_x, 'n_u': n_u,
                'train_sims': train_sims, 'valid_sims': valid_sims, 
                'test_sims': test_sims,}, model_path)
    logger.info(f'Model saved -> {model_path}  |  Training Time: {train_time:.1f} s')

    # -- Evaluate ------------------------------------------------------------------

    model.eval()

    X_va = torch.from_numpy(X_va)
    U_va = torch.from_numpy(U_va)
    valid_set_stats = evaluate_model(model, X_va, U_va, valid_sims)
    logger.info(f"Validation set {args.steps}-step NRMSE: {valid_set_stats['NRMSE'][args.steps]:.4f}")

    # -- Save per-run metrics -------------------------------------------------------

    savemat(os.path.join(metrics_dir, 'train_metrics.mat'), {
        'n_train': n_train, 'n_valid': n_valid, 'n_test': n_test,
        'training_time': np.array([train_time], dtype=np.float32),
        'valid_X_pred': valid_set_stats['X_pred'],
        'valid_errors': valid_set_stats['errors'],
        'valid_NMSE': valid_set_stats['NMSE'],
        'valid_NRMSE_sim': valid_set_stats['NRSME_sim'],
        'valid_NRMSE': valid_set_stats['NRMSE'],
        'step_NRMSE': valid_set_stats['NRMSE'][args.steps],
        **{k: np.array(v, dtype=np.float32) for k, v in history.items()}
    })
    logger.debug(f'Metrics saved -> {metrics_dir}/train_metrics.mat')