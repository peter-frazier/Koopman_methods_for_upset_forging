import torch
import torch.nn as nn


class DMDc(nn.Module):
    """
    Linearly Recurrent Autoencoder for controlled dynamical systems.
    Learns z_{k+1} = A z_k + B u_k in a nonlinear latent space.
    """

    def __init__(self, n_x, n_u):
        super().__init__()
        self.A = nn.Linear(n_x, n_x, bias=False)
        self.B = nn.Linear(n_u, n_x, bias=False)
    
    def rollout(self, x0, us):
        """
        Propagate latent state forward under control inputs.

        z0 : (batch, n_z)    initial latent state
        us : (batch, K, n_u) control sequence
        Returns list of K predicted latent states.
        """
        x = x0
        x_preds = []
        for k in range(us.shape[1]):
            x = self.A(x) + self.B(us[:, k])
            x_preds.append(x)
        return x_preds