import h5py
import os
import numpy as np

U = np.zeros((1000, 100, 1), dtype=np.float32)
X = np.zeros((1000, 101, 24807), dtype=np.float32)
alpha = np.zeros((1000, 101, 1600), dtype=np.float32)

data_file = os.path.join(os.path.dirname(__file__), '15-5PH_SS_682C_0-500.mat')
with h5py.File(data_file, 'r') as f:
    X[:500,:,:] = f['X'][:]
    U[:500,:,:] = f['U'][:]
    alpha[:500,:,:] = f['alpha'][:]

data_file = os.path.join(os.path.dirname(__file__), '15-5PH_SS_682C_500-1000.mat')
with h5py.File(data_file, 'r') as f:
    X[500:,:,:] = f['X'][:]
    U[500:,:,:] = f['U'][:]
    alpha[500:,:,:] = f['alpha'][:]

data_file = os.path.join(os.path.dirname(__file__), '15-5PH_SS_682C.mat')
with h5py.File(data_file, 'w') as f:
    f.create_dataset('X', data=X)
    f.create_dataset('U', data=U)
    f.create_dataset('alpha', data=alpha)