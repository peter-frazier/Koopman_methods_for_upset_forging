import torch
import torch.nn as nn


class DMDc(nn.Module):
    """
    Linearly Recurrent Autoencoder for controlled dynamical systems.
    Learns z_{k+1} = A z_k + B u_k in a nonlinear latent space.
    """

    def __init__(self, A, B):
        super().__init__()
        assert A.shape[0]==A.shape[1]
        assert A.shape[0]==B.shape[0]
        _, n_x = A.shape
        _, n_u = B.shape

        # A
        self.A = nn.Linear(n_x, n_x, bias=False)
        self.A.weight.data = A

        # B
        self.B = nn.Linear(n_u, n_x, bias=False)
        nn.init.xavier_uniform_(self.B.weight)
    
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