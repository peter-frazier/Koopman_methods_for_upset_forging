import torch
import torch.nn as nn


def _svd_init(n, scale):
    W = torch.randn(n, n)
    U, _, Vh = torch.linalg.svd(W)
    return (U @ Vh) * scale


class Lifting(nn.Module):
    """Learned observables Psi(x): R^n_x -> R^n_psi."""
    def __init__(self, n_x, n_psi, n_h, act, alpha):
        super().__init__()
        width = 16 * alpha

        match act:
            case 'Tanh':
                self.act = nn.Tanh()
            case 'ReLU':
                self.act = nn.ReLU()
            case 'LeakyReLU':
                self.act = nn.LeakyReLU()
            case 'Sigmoid':
                self.act = nn.Sigmoid()

        layers = []
        layers.extend([nn.Linear(n_x, width), self.act])
        for _ in range(n_h-1):
            layers.extend([nn.Linear(width, width), self.act])
        layers.extend([nn.Linear(width, n_psi)])
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class LRAN_LD(nn.Module):
    """
    LRAN with linear decoder for controlled dynamical systems.

    Latent state:  z = [x; Psi(x)]   (dimension n_z = n_x + n_psi)
    Decoder:       x_hat = z[:n_x]   (identity on the state block, exact by construction)
    Dynamics:      z_{k+1} = A z_k + B u_k
    """

    def __init__(self, n_x, n_u, n_psi, n_h, act, alpha, init_scale=0.99):
        super().__init__()
        self.n_x  = n_x
        self.n_z  = n_x + n_psi

        self.lifting = Lifting(n_x, n_psi, n_h, act, alpha)

        self.A = nn.Linear(self.n_z, self.n_z, bias=False)
        self.A.weight.data = _svd_init(self.n_z, init_scale)

        self.B = nn.Linear(n_u, self.n_z, bias=False)
        nn.init.xavier_uniform_(self.B.weight)

    def encode(self, x):
        """Lift x to z = [x, Psi(x)]."""
        return torch.cat([x, self.lifting(x)], dim=-1)

    def decode(self, z):
        """Linear decoder: return the state block (first n_x components of z)."""
        return z[..., :self.n_x]

    def rollout(self, z0, us):
        """
        Propagate latent state forward under control inputs.

        z0 : (batch, n_z)    initial lifted state
        us : (batch, K, n_u) control sequence
        Returns list of K predicted latent states.
        """
        z = z0
        z_preds = []
        for k in range(us.shape[1]):
            z = self.A(z) + self.B(us[:, k])
            z_preds.append(z)
        return z_preds
