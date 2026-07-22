import os
import numpy as onp
import h5py
from scipy.io import savemat

state_data = onp.load('jax-fem/LRAN_BLRAN/Data/state_data.npy') # Columnwise state data
U = onp.load('jax-fem/LRAN_BLRAN/Data/u_data.npy') # Columnwise input data

num_sim = 1000
num_steps = 100
Nx, _ = state_data.shape
Nu, _ = U.shape

state_data = state_data.T.reshape(num_sim, num_steps, Nx)
U = U.T.reshape(num_sim, (num_steps-1), Nu)
'''
savemat(os.path.join('jax-fem/LRAN_BLRAN/Data', 'Isothermal_Plasticity.mat'), {
    'X': state_data,
    'U': U},
    do_compression=False)
'''
with h5py.File('jax-fem/LRAN_BLRAN/Data/Isothermal_Plasticity.mat', 'w') as f:
    f.create_dataset('X', data=state_data)
    f.create_dataset('U', data=U)
