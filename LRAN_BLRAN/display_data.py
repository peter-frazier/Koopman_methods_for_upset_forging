import numpy as np
import os
import argparse
from scipy.io import loadmat
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Koopman Autoencoder Figure Generation')
parser.add_argument('--type', default='LRAN', help='model type')
parser.add_argument('--data_folder', default='metrics', help='folder containing data metrics')
args = parser.parse_args()

folder_path = os.path.join(os.path.dirname(__file__), args.type, args.data_folder)
file_path = os.path.join(folder_path, 'metrics.mat')
metrics = loadmat(file_path)

test_err_ts = metrics['test_err_ts'].squeeze()
steps = np.arange(len(test_err_ts))

loss = metrics['loss'].squeeze()
loss_id = metrics['loss_id'].squeeze()
loss_fwd = metrics['loss_fwd'].squeeze()
loss_lin = metrics['loss_lin'].squeeze()
epochs = np.arange(len(loss)) + 1

fig1, ax = plt.subplots(figsize=(9, 7))

ax.plot(steps, test_err_ts)
ax.set_xlabel('Step')
ax.set_ylabel('Relative Error')
ax.set_title("Time Series Error")

fig1.savefig(os.path.join(folder_path, 'test_err_ts.png'))

fig2, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)

axes[0].plot(epochs, loss, linewidth=2, color='black', label='Total Loss')
axes[0].set_title('Total Loss')
axes[0].set_yscale('log')
axes[1].plot(epochs, loss_id, color='red', label='Reconstruction Loss')
axes[1].set_title('Reconstruction Loss')
axes[1].set_yscale('log')
axes[2].plot(epochs, loss_fwd, color='blue', label='Forward Loss')
axes[2].set_title('Forward Loss')
axes[2].set_yscale('log')
axes[3].plot(epochs, loss_lin, color='green', label='Linear Loss')
axes[3].set_title('Linear Loss')
axes[3].set_yscale('log')
axes[3].set_xlabel('Epoch')

fig2.savefig(os.path.join(folder_path, f'{args.type} training_loss.png'))

plt.show()