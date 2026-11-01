import h5py
import os
import numpy as np

data_file = os.path.join(os.path.dirname(__file__), '15-5PH_SS_682C.mat')

print('Reading data')
with h5py.File(data_file, 'r') as f:
    X = f['X'][:] 
    U = f['U'][:]
    alpha = f['alpha'][:]

print(f'max stress: {np.max(X)}')
print(f'min stress: {np.min(X)}')
print(f'largest initial plastic strain: {np.max(alpha[:, 0, :])}')
print(f'largest plastic strain: {np.max(alpha)}')
print(f'zero: {np.min(alpha)}')

data_file = os.path.join(os.path.dirname(__file__), '15-5PH_SS_682C_0-500.mat')

print('Writing data')
with h5py.File(data_file, 'w') as f:
    f.create_dataset('X', data=X[:500,:,:])
    f.create_dataset('U', data=U[:500,:,:])
    f.create_dataset('alpha', data=alpha[:500,:,:])

data_file = os.path.join(os.path.dirname(__file__), '15-5PH_SS_682C_500-1000.mat')

print('Writing data')
with h5py.File(data_file, 'w') as f:
    f.create_dataset('X', data=X[500:,:,:])
    f.create_dataset('U', data=U[500:,:,:])
    f.create_dataset('alpha', data=alpha[500:,:,:])
