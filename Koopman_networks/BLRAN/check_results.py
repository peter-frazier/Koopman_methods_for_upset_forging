"""
Aggregates BLRAN hyperparameter sweep results and reports best hyperparameters.

Grid, seeds, and result directory naming are read directly from create_script.py
-- edit only create_script.py to change the sweep configuration.


Outputs:
    blran_sweep_results.txt      -- ranked summary of all combinations
    best_results/avg_metrics.mat -- averaged loss/error curves for the best combo
    best_results/seed{i}.mat    -- A, B, encoder, decoder weights + errors per seed
"""

import os
import numpy as np
import torch
import argparse
from scipy.io import loadmat, savemat
from create_script import create_grid, all_combos, result_dir


parser = argparse.ArgumentParser(description='BLRAN model grid checker')

parser.add_argument('--data_name',    default='Isothermal_Plasticity',
                        help='label used for data in folder')
parser.add_argument('--mode',         default='sweep',
                        help='data ablation, hyperparameter sweep, or cross-validation')
parser.add_argument('--n_seeds', type = int, default=1,
                        help='number of seeds in sweep')

args = parser.parse_args()

def _load_metrics(path):
    raw = loadmat(path)
    return {k: np.squeeze(v) for k, v in raw.items() if not k.startswith('__')}


# -- Load and average across seeds ---------------------------------------------
GRID, FIXED, SEEDS = create_grid(args.mode)
records   = []
n_missing = 0
n_nan     = 0

for p in all_combos():
    per_seed = []
    for seed in SEEDS:
        path = os.path.join(result_dir(p, seed), 'train_metrics.mat')
        if not os.path.isfile(path):
            n_missing += 1
            continue
        d = _load_metrics(path)
        if not np.isfinite(float(d['step_NRMSE'])):
            n_nan += 1
            continue
        per_seed.append(d)

    if not per_seed:
        continue

    avg = {k: np.mean(np.stack([d[k] for d in per_seed]), axis=0)
           for k in per_seed[0]}
    records.append({'params': p, 'n_seeds': len(per_seed), **avg})

if n_missing:
    print(f'Warning: {n_missing} result file(s) missing (jobs still running or failed).')
if n_nan:
    print(f'Warning: {n_nan} seed(s) skipped due to NaN/Inf error (diverged runs).')

if not records:
    print('No results found. Run jobs first.')
    raise SystemExit(1)

# -- Rank combinations ----------------------------------------------------------
records.sort(key=lambda r: float(r['step_NRMSE']))
best      = records[0]
min_err   = float(best['step_NRMSE'])
near_best = [r for r in records if float(r['step_NRMSE']) <= 1.2 * min_err]

# -- Summary table helpers ------------------------------------------------------
_GRID_KEYS = list(GRID.keys())
_SEP = '-' * (14 + 12 * len(_GRID_KEYS))

def _hdr():
    cols = '  '.join(f'{k:>10}' for k in _GRID_KEYS)
    return f'{"rank":>4}  {"err":>10}  {cols}  {"seeds":>5}'

def _row(rank, r):
    p    = r['params']
    cols = '  '.join(f'{p[k]:>10}' for k in _GRID_KEYS)
    return f'{rank:4d}  {float(r["step_NRMSE"]):10.4f}  {cols}  {r["n_seeds"]:5d}'

# -- Write summary --------------------------------------------------------------
summary_path = f'blran_{args.mode}_sweep_results.txt'
with open(summary_path, 'w') as fh:
    fh.write(f'BLRAN hyperparameter sweep -- {len(records)} combinations evaluated\n')
    fh.write('=' * len(_SEP) + '\n\n')

    fh.write(f'Best combination (lowest avg error, {best["n_seeds"]} seeds):\n')
    for k, v in best['params'].items():
        fh.write(f'  {k:12s} = {v}\n')
    fh.write(f'  step_NRMSE = {min_err:.4f}\n')
    fh.write(f'  elapsed_time  = {float(best["elapsed_time"]):.1f} s\n\n')

    fh.write(f'Top {len(near_best)} within 20% of best error:\n')
    fh.write(_hdr() + '\n' + _SEP + '\n')
    for rank, r in enumerate(near_best, 1):
        fh.write(_row(rank, r) + '\n')

    fh.write('\nAll combinations (sorted by error):\n')
    fh.write(_hdr() + '\n' + _SEP + '\n')
    for rank, r in enumerate(records, 1):
        fh.write(_row(rank, r) + '\n')

print(f'Summary -> {summary_path}')
print(f'Best:  err={min_err:.4f}  params={best["params"]}')

# -- Save best_results/ ---------------------------------------------------------
os.makedirs('best_results', exist_ok=True)

# Averaged metrics across seeds
savemat('best_results/avg_metrics.mat', {
    'step_NRMSE': np.atleast_1d(best['step_NRMSE']).astype(np.float32),
    'elapsed_time':  np.atleast_1d(best['elapsed_time']).astype(np.float32),
    'loss':          np.atleast_1d(best['loss']).astype(np.float32),
    'loss_id':       np.atleast_1d(best['loss_id']).astype(np.float32),
    'loss_fwd':      np.atleast_1d(best['loss_fwd']).astype(np.float32),
    'loss_lin':      np.atleast_1d(best['loss_lin']).astype(np.float32),
    'loss_eig':      np.atleast_1d(best['loss_eig']).astype(np.float32),
})

# Per-seed model weights and errors
for seed in SEEDS:
    rdir      = result_dir(best['params'], seed)
    model_path   = os.path.join(rdir, 'model.pt')
    metrics_path = os.path.join(rdir, 'metrics.mat')
    if not os.path.isfile(model_path) or not os.path.isfile(metrics_path):
        continue

    ckpt = torch.load(model_path, map_location='cpu')
    sd   = ckpt['state_dict']

    seed_data = {
        'A': sd['A.weight'].numpy(),
        'B': sd['B.weight'].numpy(),
        **{k.replace('.', '_'): v.numpy()
           for k, v in sd.items()
           if k.startswith('encoder') or k.startswith('decoder')},
        **{k: np.atleast_1d(v).astype(np.float32)
           for k, v in _load_metrics(metrics_path).items()},
    }
    savemat(f'best_results/seed{seed}.mat', seed_data)

print('Best results -> best_results/')
