import numpy as np


# -- Normalization -------------------------------------------------------------

def normalize(X, U):
    """
    Min-max normalize X and U to [-1, 1] per feature.

    Returns normalized arrays and a scale dict needed for denormalization.
    """
    def _minmax(arr):
        flat = arr.reshape(-1, arr.shape[-1])
        lo   = flat.min(axis=0)
        rng  = np.ptp(flat, axis=0)
        rng[rng == 0] = 1.0
        return (2 * (arr - lo) / rng - 1).astype(np.float32), lo, rng

    x_max = np.ones((24807,), dtype=np.float32)

    strain_idx = [list(range(j,j+6)) for j in [12*i for i in range(1600)]]
    strain_idx = [i for index_set in strain_idx for i in index_set]
    max_strain = np.max(np.abs(X[:,:,strain_idx]))
    x_max[strain_idx] *= max_strain

    stress_idx = [list(range(j+6,j+12)) for j in [12*i for i in range(1600)]]
    stress_idx = [i for index_set in stress_idx for i in index_set]
    max_stress = np.max(np.abs(X[:,:,stress_idx]))
    x_max[stress_idx] *= max_stress

    disp_idx = [list(range(12*1600, 24807))]
    max_disp = np.max(np.abs(X[:,:,disp_idx]))
    x_max[disp_idx] *= max_disp

    x_lo = -x_max
    x_rng = 2*x_max
    X -= x_lo
    X *= 2/x_rng
    X -=1
    X_n = X

    U_n, u_lo, u_rng = _minmax(U)

    scale = dict(x_lo=x_lo, x_rng=x_rng, u_lo=u_lo, u_rng=u_rng)
    return X_n, U_n, scale

def denormalize(arr, lo, rng):
    return (arr + 1) / 2 * rng + lo