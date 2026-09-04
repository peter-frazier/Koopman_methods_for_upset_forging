import torch
import torch.nn as nn


def _svd_init(n, scale):
    W = torch.randn(n, n)
    U, _, Vh = torch.linalg.svd(W)
    return (U @ Vh) * scale


class Encoder(nn.Module):
    def __init__(self, n_x, n_z, n_h, act, alpha):
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
        layers.extend([nn.Linear(width, n_z)])
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class BLRAN_LD(nn.Module):
    """
    Bilinear Recurrent Autoencoder with linear decoder.

    Latent state:  z = [x; Psi(x)]   (dimension n_z = n_x + n_psi)
    Decoder:       x_hat = z[:n_x]   (identity on the state block, exact by construction)
    Dynamics:      z_{k+1} = A z_k + B (z_k (x) u_k)

    (x) denotes the Kronecker product of column vectors z_k and u_k,
    giving a vector of length n_z * n_u. B has shape (n_z, n_z * n_u).
    """

    def __init__(self, n_x, n_u, n_z, n_h, act, alpha, init_scale=0.99):
        super().__init__()
        self.n_x  = n_x
        self.n_z  = n_z
        self.n_u  = n_u
        self.encoder = Encoder(n_x, n_z, n_h, act, alpha)

        # A: orthogonal init scaled to spectral radius ~ init_scale
        self.A = nn.Linear(self.n_z, self.n_z, bias=False)
        self.A.weight.data = _svd_init(self.n_z, init_scale)

        # B: maps the bilinear term z (x) u (length n_z * n_u) to latent space
        self.B = nn.Linear(self.n_z * n_u, self.n_z, bias=False)
        nn.init.xavier_uniform_(self.B.weight)

        # C
        self.C = nn.Linear(n_z, n_x, bias=False)
        nn.init.xavier_uniform_(self.B.weight)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.C(z)

    def _kron(self, z, u):
        """
        Kronecker product of column vectors z and u, batched.

        z : (batch, n_z)
        u : (batch, n_u)
        Returns (batch, n_z * n_u) -- equivalent to flattening the outer
        product z u^T row by row for each sample.
        """
        return torch.einsum('bi,bj->bij', z, u).reshape(z.shape[0], self.n_z * self.n_u)

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
            z = self.A(z) + self.B(self._kron(z, us[:, k]))
            z_preds.append(z)
        return z_preds
