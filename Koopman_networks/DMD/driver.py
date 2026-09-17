import sys
sys.path.insert(0, "/users/PAS3353/peterfrazier/Koopman_methods_for_upset_forging") # TODO: change to folder containing jax-fem-checkpoint

import argparse
import os
import jax
import time
import h5py
import json
import numpy as np
import jax.numpy as jnp
from scipy.io import savemat
from jax_fem_checkpoint import logger

from read_dataset import normalize


def evaluate_model(A, B, X, U, sims):
    '''
    INPUTS
    model: PyTorch model to be evaluated
    X: state dataset
    U: input dataset
    sims: simulated trajectory number
    device: run sims on cpu or gpu

    OUTPUTS
    X_pred: predicted states of X (n_traj, traj_len, n_x)        
    errors: errors between X and X_pred (n_traj, traj_len, n_x)
    RE: relative error per-sim (n_traj, traj_len)
    NRMSE_sim: running NMRSE per-sim (n_traj, traj_len)
    NRMSE: running NMRSE across dataset (traj_len,)
    times: list of simulation runtimes
    '''

    @jax.jit
    def linear_simulation(A, B, x0, us):
        x_hat = jnp.zeros(shape=(x0.shape[0], us.shape[1]+1), dtype=jnp.float32)
        x_hat = x_hat.at[:, 0].set(x0)
        x_now = x0

        for i, disp in enumerate(us.T):
            x_next = A@x_now + B@disp
            x_hat = x_hat.at[:,i+1].set(x_next)
            x_now = x_next

        return x_hat

    n_traj, traj_len, n_x = X.shape
    X_pred    = np.zeros(shape=X.shape, dtype=np.float32)
    times = []

    A = jnp.asarray(A)
    B = jnp.asarray(B)

    for i, sim in enumerate(sims):
        start = time.time()

        x0 = jnp.asarray(X[i, :1, :].T) # (n_x, 1)
        us = jnp.asarray(U[i, : , :].T) # (1, traj_len - 1)

        x_hat = linear_simulation(A, B, x0, us)

        end = time.time()
        logger.debug(f'Ran linear simulation {i+1} of {len(sims)} in {end-start} seconds')
        times.append(end-start)

        X_pred[i, :, :] = x_hat

    X = X
    errors = X - X_pred
    RE = np.linalg.norm(errors, axis=2)/np.linalg.norm(X, axis=2)

    NRMSE_sim = np.zeros(shape=(n_traj, traj_len), dtype=np.float32)
    NRMSE     = np.zeros(shape=(traj_len,), dtype=np.float32)
    for i in range(traj_len):
        vnorm = np.linalg.norm(X[:,:i+1,:], axis=2)      #(n_traj, i)
        enorm = np.linalg.norm(errors[:,:i+1,:], axis=2) #(n_traj, i)
        NRMSE_sim[:, i] = (np.sum(enorm**2, axis=1)/np.sum(vnorm**2, axis=1))**0.5 #(n_traj,)
        NRMSE[i]        = (np.sum(enorm**2)/np.sum(vnorm**2))**0.5 #(,)

    run_stats = {
        'X_pred': X_pred,
        'errors': errors,
        'RE': RE,
        'NRMSE_sim': NRMSE_sim,
        'NRMSE': NRMSE,
        'times': np.array(times, dtype=np.float32)
    }
    return run_stats


if __name__ == "__main__":

    # -- Arguments -----------------------------------------------------------------

    parser = argparse.ArgumentParser(description='DMDc -- controlled dynamical systems')

    # Data
    parser.add_argument('--data_name',    default='Isothermal_Plasticity',
                        help='label used for data in folder')
    parser.add_argument('--train_num', type=int, default=210,
                        help='number of trajectories used for training')
    parser.add_argument('--valid_num', type=int, default=30,
                        help='number of trajectories used for validation')
    parser.add_argument('--test_num',  type=int, default=60,
                        help='number of trajectories used for testing')
    parser.add_argument('--shift_frac', type=float, default=0.,
                        help='fraction of trajectories to shift to get different sets')

    parser.add_argument('--seed',       type=int,   default=0)
    parser.add_argument('--device',     type=str,   default='cpu')
    parser.add_argument('--out_dir',    type=str,   default='metrics',
                        help='directory to save per-run metrics (empty = skip)')

    args = parser.parse_args()

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
    n_train = args.train_num
    n_valid = args.valid_num
    n_test  = args.test_num
    n_shift = int(n_traj*args.shift_frac)
    sims    = np.roll(sims, shift=n_shift)

    test_sims  = sims[:n_test]
    train_sims = sims[n_test:n_test+n_train]
    valid_sims = sims[n_test+n_train:n_test+n_train+n_valid]
    X_tr, U_tr = X_n[train_sims,:,:], U_n[train_sims,:,:]
    X_va, U_va = X_n[valid_sims,:,:], U_n[valid_sims,:,:]

    # -- Train ---------------------------------------------------------------------

    # Implementation of Koopman lifting
    def lift_function(data):
        lift_data = data # identity lifting function
        return lift_data

    def dataset_for_DMD(arr):
        _, _, dim = arr.shape
        arr = np.transpose(arr) # dim, traj_length - 1, n_traj
        arr = np.reshape(arr, shape=(dim, -1), order='C')
        return arr

    def DMDc_model(X, Y, U):
        n_traj, _, n_x = X.shape # n_traj, traj_length - 1, dim

        logger.info('Lifting data...')
        X_lift = lift_function(X)
        Y_lift = lift_function(Y)
        n_z, _ = X_lift.shape

        X_lift = dataset_for_DMD(X_lift)
        Y_lift = dataset_for_DMD(Y_lift)
        U      = dataset_for_DMD(U)

        logger.info('Condensing data...')
        G = np.vstack((X_lift, U))@np.vstack((X_lift, U)).T
        V = Y_lift@np.vstack((X_lift, U)).T

        logger.info('Finding best-fit operators')
        M = V@np.linalg.pinv(G, rcond=1e-6)

        A = M[:, :n_z]
        B = M[:, n_z:]

        return A, B

    t0 = time.time()

    X_dict = X_tr[train_sims, :-1, :]
    Y_dict = X_tr[train_sims, 1:, :]
    U_dict = U_tr[train_sims, :, :]
    A, B = DMDc_model(X_dict, Y_dict, U_dict)

    train_time = time.time() - t0

    metrics_dir = os.path.join(os.path.dirname(__file__), args.out_dir)
    os.makedirs(metrics_dir, exist_ok=True)
    model_path = os.path.join(metrics_dir, 'model.npz')
    np.savez(model_path, A=A, B=B, n_x=n_x, n_u=n_u,
             train_sims=train_sims, valid_sims=valid_sims, test_sims=test_sims)
    logger.info(f'Model saved -> {model_path}  |  Training Time: {train_time:.1f} s')

    # -- Evaluate ------------------------------------------------------------------

    valid_set_stats = evaluate_model(A, B, X_va, U_va, valid_sims)
    logger.info(f"Validation set {args.steps}-step NRMSE: {valid_set_stats['NRMSE'][args.steps]:.4f}")

    # -- Save per-run metrics -------------------------------------------------------
    
    savemat(os.path.join(metrics_dir, 'train_metrics.mat'), {
        'n_train': n_train, 'n_valid': n_valid, 'n_test': n_test,
        'train_sims': train_sims,
        'valid_sims': valid_sims,
        'test_sims': test_sims,
        'training_time': np.array([train_time], dtype=np.float32),
        'valid_RE': valid_set_stats['RE'],
        'valid_NRMSE_sim': valid_set_stats['NRMSE_sim'],
        'valid_NRMSE': valid_set_stats['NRMSE'],
        'step_NRMSE': valid_set_stats['NRMSE'][args.steps]}
    )
    logger.debug(f'Metrics saved -> {metrics_dir}/train_metrics.mat')
