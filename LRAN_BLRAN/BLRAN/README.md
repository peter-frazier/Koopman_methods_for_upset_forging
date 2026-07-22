# BLRAN : Bilinear Recurrent Autoencoder

BLRAN replaces the linear control term in LRAN with a bilinear interaction between the latent state and the control input:

```
z_{k+1} = A z_k + B (z_k (x) u_k)
```

where (x) denotes the Kronecker product of column vectors z_k (dimension n_z) and u_k (dimension n_u), producing a vector of length n_z * n_u. B therefore has shape (n_z, n_z * n_u).

The Kronecker product z_k (x) u_k is computed as the outer product z_k u_k^T (shape n_z x n_u) flattened row by row, which gives every pairwise product z_i * u_j as a single vector.

The encoder and decoder are the same nonlinear MLPs as in LRAN. All three training losses (L_id, L_fwd, L_lin) and the optional eigenvalue penalty on A (L_eig) are identical to LRAN.


---

## Files

| File | Description |
|---|---|
| `model.py` | Encoder, Decoder, and BLRAN model with Kronecker rollout |
| `train.py` | Training loop (reconstruction + forward + linearity losses) |
| `read_dataset.py` | Data loading, normalization, windowing |
| `driver.py` | Main script: load data, train, evaluate, save |
| `create_script.py` | Generates SLURM job scripts for a hyperparameter sweep |
| `check_results.py` | Aggregates sweep results and saves best model parameters |

---

## Dataset Requirements

The data should be a `.mat` file with two variables:

- **`X`**: state snapshots, shape `(n_traj, T, n_x)` or `(T, n_x)` for a single trajectory
- **`U`**: control inputs, shape `(n_traj, T-1, n_u)` or `(T-1, n_u)`

`T` is the number of state snapshots; there are `T-1` control steps between them. States can be anything -- discretized PDE fields, modal coefficients, sensor readings, etc. BLRAN normalizes all features to `[-1, 1]` internally.

If the `.mat` file uses different variable names, pass `--x_key` and `--u_key`.

---

## Quick Start

**Pendulum system (for testing):**
```bash
python driver.py --dataset pendulum --epochs 500 --n_z 8
```

**Custom dataset:**
```bash
python driver.py --dataset /path/to/data.mat --x_key X --u_key U \
    --n_z 16 --alpha 4 --steps 8 --epochs 500 --out_dir results/run1
```

Key arguments:

| Argument | Description |
|---|---|
| `--n_z` | Latent dimension |
| `--alpha` | Width multiplier (hidden layer width = 16 x alpha) |
| `--steps` | Multi-step prediction horizon during training |
| `--epochs` | Training epochs |
| `--lr` | Learning rate |
| `--gamma_id` | Weight on reconstruction loss |
| `--gamma_fwd` | Weight on decoded forward-prediction loss |
| `--gamma_lin` | Weight on latent linearity loss |
| `--gamma_eig` | Weight on eigenvalue stability penalty (0 = off) |
| `--out_dir` | Directory to save `model.pt` and `metrics.mat` |
| `--no_plot` | Suppress figures (use on HPC) |

---

## Hyperparameter Sweep (SLURM / OSC)

1. **Edit `create_script.py`**: set `DATASET`, `GRID`, `SEEDS`, `FIXED`, and SLURM settings (`ACCOUNT`, `CONDA`, etc.).
2. **Generate job scripts:**
   ```bash
   python create_script.py
   ```
   Writes one `.sh` per hyperparameter-seed combination and a `submit_batch1.sh`.
3. **Submit on the cluster:**
   ```bash
   sbatch submit_batch1.sh
   ```
   One email when the submission batch starts/ends/fails. Individual jobs run silently.
4. **Aggregate results:**
   ```bash
   python check_results.py
   ```
   Outputs:
   - `blran_sweep_results.txt`: all combinations ranked by mean test error
   - `best_results/avg_metrics.mat`: loss and error curves averaged across seeds for the best combo
   - `best_results/seed{i}.mat`: per-seed `A`, `B`, encoder/decoder weights, and test errors

---

## Outputs

Each run with `--out_dir` produces:

- **`model.pt`**: full checkpoint (state dict, args, normalization scale)
- **`metrics.mat`**: training loss curves (`loss`, `loss_id`, `loss_fwd`, `loss_lin`, `loss_eig`), test error time series (`test_err_ts`), mean test error (`test_err_mean`), and wall time

The `.mat` files are loadable in MATLAB and Python (`scipy.io.loadmat`).

---

## Training Losses

BLRAN uses the same three losses as LRAN, applied over a window of K+1 consecutive states and K controls:

- **L_id**: encode then decode each state (autoencoder reconstruction)
- **L_fwd**: roll out the bilinear latent dynamics from z_t and decode; compare to true states
- **L_lin**: roll out the bilinear latent dynamics from z_t; compare to encoded ground truth

The eigenvalue penalty L_eig acts on A only (the linear part of the dynamics).

---

## Comparison with LRAN

| | LRAN | BLRAN |
|---|---|---|
| Latent dynamics | A z_k + B u_k | A z_k + B (z_k (x) u_k) |
| B shape | (n_z, n_u) | (n_z, n_z * n_u) |
| Captures state-control coupling | no | yes |
| Encoder / Decoder | nonlinear MLP | nonlinear MLP (same) |
| Training losses | L_id, L_fwd, L_lin, L_eig | L_id, L_fwd, L_lin, L_eig (same) |
