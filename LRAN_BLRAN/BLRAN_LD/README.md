# BLRAN-LD : Bilinear Recurrent Autoencoder with Linear Decoder

BLRAN-LD combines the bilinear latent dynamics from BLRAN with a linear decoder:

```
z = [x; Psi(x)]                      (lifting, dimension n_z = n_x + n_psi)
x_hat = [I, 0] z = z[:n_x]           (linear decoder, exact by construction)
z_{k+1} = A z_k + B (z_k (x) u_k)   (bilinear dynamics)
```

(x) denotes the Kronecker product of column vectors z_k and u_k, giving a vector of length n_z * n_u. B therefore has shape (n_z, n_z * n_u).

Only Psi(x) and the matrices A and B are learned. Because the decoder is exact by construction, the reconstruction loss (L_id) is always zero and is dropped from training.

**Note on B size:** B grows as n_z * n_u where n_z = n_x + n_psi, so for systems with large state or control dimension the bilinear parameter count can be significant.

---

## Files

| File | Description |
|---|---|
| `model.py` | Lifting network, BLRAN_LD model with encode/decode/Kronecker rollout |
| `train.py` | Training loop (forward prediction + linearity losses) |
| `read_dataset.py` | Data loading, normalization, windowing |
| `driver.py` | Main script: load data, train, evaluate, save |
| `create_script.py` | Generates SLURM job scripts for a hyperparameter sweep |
| `check_results.py` | Aggregates sweep results and saves best model parameters |

---

## Dataset Requirements

The data should be a `.mat` file with two variables:

- **`X`**: state snapshots, shape `(n_traj, T, n_x)` or `(T, n_x)` for a single trajectory
- **`U`**: control inputs, shape `(n_traj, T-1, n_u)` or `(T-1, n_u)`

`T` is the number of state snapshots; there are `T-1` control steps between them. States can be anything -- discretized PDE fields, modal coefficients, sensor readings, etc. BLRAN-LD normalizes all features to `[-1, 1]` internally.

If the `.mat` file uses different variable names, pass `--x_key` and `--u_key`.

---

## Quick Start

**Pendulum system (for testing):**
```bash
python driver.py --dataset pendulum --epochs 500 --n_psi 6
```

**Custom dataset:**
```bash
python driver.py --dataset /path/to/data.mat --x_key X --u_key U \
    --n_psi 16 --alpha 4 --steps 8 --epochs 500 --out_dir results/run1
```

Key arguments:

| Argument | Description |
|---|---|
| `--n_psi` | Number of learned observables (total latent dim = n_x + n_psi) |
| `--alpha` | Width multiplier (hidden layer width = 16 x alpha) |
| `--steps` | Multi-step prediction horizon during training |
| `--epochs` | Training epochs |
| `--lr` | Learning rate |
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
   - `blran_ld_sweep_results.txt`: all combinations ranked by mean test error
   - `best_results/avg_metrics.mat`: loss and error curves averaged across seeds for the best combo
   - `best_results/seed{i}.mat`: per-seed `A`, `B`, lifting network weights, and test errors

---

## Outputs

Each run with `--out_dir` produces:

- **`model.pt`**: full checkpoint (state dict, args, normalization scale)
- **`metrics.mat`**: training loss curves (`loss`, `loss_fwd`, `loss_lin`, `loss_eig`), test error time series (`test_err_ts`), mean test error (`test_err_mean`), and wall time

The `.mat` files are loadable in MATLAB and Python (`scipy.io.loadmat`).

---

## Training Losses

BLRAN-LD is trained with two losses over a window of K+1 consecutive states and K controls.
Let z_k = [x_k; Psi(x_k)] denote the lifted state.

- **L_fwd**: roll out the bilinear latent dynamics from z_t; decode (take first n_x components) and compare to true states x_{t+1}, ..., x_{t+K}
- **L_lin**: roll out the bilinear latent dynamics from z_t; compare the full predicted latent vectors to the lifted ground truth z_{t+1}, ..., z_{t+K}

L_lin enforces that Psi evolves consistently with the learned bilinear model, not just the state block. The eigenvalue penalty L_eig acts on A only.

---

## Comparison with Other Models

| | LRAN | LRAN-LD | BLRAN | BLRAN-LD |
|---|---|---|---|---|
| Latent state | z (free) | [x; Psi(x)] | z (free) | [x; Psi(x)] |
| Decoder | learned MLP | fixed [I, 0] | learned MLP | fixed [I, 0] |
| Dynamics | Az + Bu | Az + Bu | Az + B(z(x)u) | Az + B(z(x)u) |
| B shape | (n_z, n_u) | (n_z, n_u) | (n_z, n_z*n_u) | (n_z, n_z*n_u) |
| L_id | yes | no | yes | no |
| Latent dim arg | --n_z | --n_psi | --n_z | --n_psi |
