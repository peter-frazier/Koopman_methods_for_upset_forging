import argparse
import os
import time
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from scipy.io import savemat
from jax_fem_checkpoint import logger
import matplotlib.pyplot as plt

from model import BLRAN
from train import train as train_blran
from read_dataset import (generate_pendulum, load_from_mat,
                           normalize, denormalize, make_windows)

# -- Arguments -----------------------------------------------------------------
parser = argparse.ArgumentParser(description='BLRAN -- Bilinear Recurrent Autoencoder')

# Data
parser.add_argument('--data_folder',default=os.path.join(os.path.dirname((os.path.dirname(__file__))), 'Data'),
                    help='folder containing .mat file of data')
parser.add_argument('--dataset',    default='pendulum',
                    help='"pendulum" or path to a .mat file')
parser.add_argument('--x_key',      default='X',   help='key for states in .mat file')
parser.add_argument('--u_key',      default='U',   help='key for controls in .mat file')
parser.add_argument('--n_traj',     type=int,   default=100,
                    help='number of trajectories to generate (pendulum only)')
parser.add_argument('--traj_len',   type=int,   default=200,
                    help='number of control steps per trajectory (pendulum only)')
parser.add_argument('--train_frac', type=float, default=0.8,
                    help='fraction of trajectories used for training')

# Architecture
parser.add_argument('--n_z',        type=int,   default=8,
                    help='latent (Koopman) dimension')
parser.add_argument('--n_h',        type=int,   default=2,
                    help='number of hidden layers')
parser.add_argument('--activation', type=str,   default='Tanh',
                    help='network width multiplier (hidden layer width = 16*alpha)')
parser.add_argument('--alpha',      type=int,   default=4,
                    help='network width multiplier (hidden layer width = 16*alpha)')
parser.add_argument('--init_scale', type=float, default=0.99,
                    help='initial spectral radius of A')

# Training
parser.add_argument('--steps',      type=int,   default=8,
                    help='multi-step prediction horizon during training')
parser.add_argument('--epochs',     type=int,   default=500)
parser.add_argument('--batch_size', type=int,   default=128)
parser.add_argument('--lr',         type=float, default=1e-3)
parser.add_argument('--wd',         type=float, default=1e-4)
parser.add_argument('--gradclip',   type=float, default=0.05)
parser.add_argument('--gamma_id',   type=float, default=1.0)
parser.add_argument('--gamma_fwd',  type=float, default=1.0)
parser.add_argument('--gamma_lin',  type=float, default=1.0)
parser.add_argument('--gamma_eig',  type=float, default=0.0,
                    help='weight on eigenvalue stability loss (0 = disabled)')

parser.add_argument('--seed',       type=int,   default=0)
parser.add_argument('--device',     type=str,   default='cpu')
parser.add_argument('--out_dir',    type=str,   default=os.path.join(os.path.dirname(__file__), 'metrics'),
                    help='directory to save per-run metrics (empty = skip)')
parser.add_argument('--no_plot',    action='store_true',
                    help='suppress figure output (use on HPC without display)')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# -- Data ----------------------------------------------------------------------
if args.dataset == 'pendulum':
    logger.debug('Generating data...')
    X, U = generate_pendulum(n_traj=args.n_traj, T=args.traj_len, seed=args.seed)
elif args.dataset == 'Isothermal_Plasticity.mat':
    logger.debug("Loading data...")
    file = os.path.join(args.data_folder, args.dataset)

    # Load necessary info
    with open(os.path.join(args.data_folder, 'Simulation_Info.json'), 'r') as f:
        sim_info = json.load(f)
    num_sim = sim_info['num_sim']
    num_steps = sim_info['num_steps']
    train_sims = np.array(sim_info['train_idx'])
    test_sims = np.array(sim_info['test_idx'])
    with h5py.File(file, 'r') as f:
        X = f[args.x_key][:] 
        U = f[args.u_key][:]
else:
    X, U = load_from_mat(os.path.join(args.data_folder, args.dataset), x_key=args.x_key, u_key=args.u_key)

_, _, n_x = X.shape
_, _, n_u = U.shape

logger.debug('Normalizing data on [-1, 1]...')
X_n, U_n, scale = normalize(X, U, args.dataset)

logger.debug('Splitting testing/training sets...')
if args.dataset == 'Isothermal_Plasticity.mat':
    n_train    = int(len(train_sims)*args.train_frac/0.8)
    train_sims = train_sims[:n_train]
    X_tr, U_tr = X_n[train_sims,:,:], U_n[train_sims,:,:]
    X_te, U_te = X_n[test_sims,:,:] , U_n[test_sims,:,:]
else:  
    n_train      = int(args.train_frac * len(X_n))
    X_tr, U_tr   = X_n[:n_train], U_n[:n_train]
    X_te, U_te   = X_n[n_train:], U_n[n_train:]

logger.debug('Forming dataset for PyTorch...')
if args.dataset == 'Isothermal_Plasticity.mat':
    class WindowDataset(Dataset):
        def __init__(self, X, U, steps):
            self.X = X # (n_traj, T, n_x)
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
        
    X_tr = torch.from_numpy(X_tr)
    U_tr = torch.from_numpy(U_tr)
    dataset = WindowDataset(X_tr, U_tr, args.steps)
else:
    windows = make_windows(X_tr, U_tr, args.steps)
    dataset = TensorDataset(*[torch.from_numpy(w) for w in windows])

loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

logger.info(f'Train trajectories: {n_train}  |  Test trajectories: {len(X_te)}')
logger.info(f'Windows per epoch:  {len(dataset)}  |  Batches per epoch: {len(loader)}')

# -- Model ---------------------------------------------------------------------
model   = BLRAN(n_x, n_u, args.n_z, args.n_h, args.activation, args.alpha, args.init_scale)
n_param = sum(p.numel() for p in model.parameters())
logger.info(f'BLRAN | n_x={n_x}  n_u={n_u}  n_z={args.n_z}  n_h={args.n_h}  alpha={args.alpha} '
      f'(width={16*args.alpha})  activation={args.activation} | B shape=({args.n_z}, {args.n_z * n_u}) | params={n_param:,}')

# -- Train ---------------------------------------------------------------------
t0 = time.time()
history = train_blran(
    model, loader, args.epochs,
    lr=args.lr, wd=args.wd, gradclip=args.gradclip,
    gamma_id=args.gamma_id, gamma_fwd=args.gamma_fwd,
    gamma_lin=args.gamma_lin, gamma_eig=args.gamma_eig,
    device=args.device, print_every=10,
)
elapsed = time.time() - t0

if args.out_dir:
    os.makedirs(args.out_dir, exist_ok=True)
model_path = os.path.join(args.out_dir, 'model.pt') if args.out_dir else 'blran_model.pt'
torch.save({'state_dict': model.state_dict(), 'args': vars(args),
            'scale': scale, 'n_x': n_x, 'n_u': n_u}, model_path)
logger.info(f'Model saved -> {model_path}')

# -- Evaluate: roll out a test trajectory from its initial condition -----------
model.eval()
with torch.no_grad():
    x0      = torch.from_numpy(X_te[0, :1])           # (1, n_x)
    us      = torch.from_numpy(U_te[0]).unsqueeze(0)  # (1, T, n_u)
    z0      = model.encoder(x0)
    z_preds = model.rollout(z0, us)
    x_hat   = torch.cat(
        [x0] + [model.decoder(z) for z in z_preds], dim=0
    ).numpy()                                          # (T+1, n_x)

x_true = X_te[0]                                      # (T+1, n_x) normalized
T_eval  = min(len(x_hat), len(x_true))
x_hat   = x_hat[:T_eval]
x_true  = x_true[:T_eval]

err = (np.linalg.norm(x_hat - x_true, axis=1) /
       (np.linalg.norm(x_true, axis=1) + 1e-8))

logger.info(f'Mean relative error: {err.mean():.4f}  |  Elapsed: {elapsed:.1f} s')

# -- Save per-run metrics ------------------------------------------------------
if args.out_dir:
    savemat(os.path.join(args.out_dir, 'metrics.mat'), {
        'test_err_mean': np.array([err.mean()], dtype=np.float32),
        'test_err_ts':   err.astype(np.float32),
        'elapsed_time':  np.array([elapsed],    dtype=np.float32),
        **{k: np.array(v, dtype=np.float32) for k, v in history.items()}
    })
    logger.debug(f'Metrics saved -> {args.out_dir}/metrics.mat')

# -- Plots ---------------------------------------------------------------------
if not args.no_plot:
    x_hat_phys  = denormalize(x_hat,  scale['x_lo'], scale['x_rng'])
    x_true_phys = denormalize(x_true, scale['x_lo'], scale['x_rng'])

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(x_true_phys[:, 0], label='true')
    axes[0].plot(x_hat_phys[:, 0], '--', label='BLRAN')
    axes[0].set_ylabel('theta (rad)'); axes[0].legend()
    axes[0].set_title("theta'' = -(g/l) sin(theta) + u")

    axes[1].plot(x_true_phys[:, 1], label='true')
    axes[1].plot(x_hat_phys[:, 1], '--', label='BLRAN')
    axes[1].set_ylabel("theta' (rad/s)"); axes[1].legend()

    axes[2].semilogy(err)
    axes[2].set_ylabel('relative error'); axes[2].set_xlabel('time step')

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'blran_prediction.png'), dpi=150)
    plt.show()
