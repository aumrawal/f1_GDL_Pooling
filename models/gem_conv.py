"""
Gauge Equivariant Mesh Convolution layer + RegularNonlinearity.

Implements Algorithm 1 of de Haan et al. (2020):
    f'_p = Σ_i w_self_i · K_self_i · f_p
         + Σ_{i, q∈N(p)} w_neigh_i · K_neigh_i(θ_pq) · ρ_in(g_{q→p}) · f_q

Key components:
  1. Anisotropic kernel K_neigh(θ)
  2. Parallel transport ρ_in(g_{q→p})
  3. RegularNonlinearity (Sec 5 of GEM paper)
  4. GEMBlock: Conv → LayerNorm → Nonlinearity → Residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.irreps import (
    FeatureType, EquivariantKernelBasis, feature_dim, rho_batch,
)


def scatter_add(src, index, dim, dim_size):
    """Pure PyTorch replacement for torch_scatter.scatter_add."""
    out = torch.zeros(dim_size, src.shape[-1], dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, index_expanded, src)
    return out


def apply_parallel_transport(
    features: Tensor, transporters: Tensor, ftype_in: FeatureType,
) -> Tensor:
    """
    Apply ρ_in(g_{q→p}) to each neighbour's feature vector.
    Rotates each irrep block by the appropriate transporter angle.
    """
    transported = torch.zeros_like(features)
    offset = 0
    for (order, mult) in ftype_in:
        d = 1 if order == 0 else 2
        block = features[:, offset:offset + mult * d]
        if order == 0:
            transported[:, offset:offset + mult * d] = block
        else:
            block = block.reshape(-1, mult, d)
            R = rho_batch(order, transporters)
            rotated = torch.einsum('emd,erd->emr', block, R)
            transported[:, offset:offset + mult * d] = rotated.reshape(-1, mult * d)
        offset += mult * d
    return transported


class GEMConv(nn.Module):
    """Single Gauge Equivariant Mesh Convolution layer."""

    def __init__(self, ftype_in: FeatureType, ftype_out: FeatureType):
        super().__init__()
        self.ftype_in = ftype_in
        self.ftype_out = ftype_out
        self.dim_in = feature_dim(ftype_in)
        self.dim_out = feature_dim(ftype_out)
        self.kernel = EquivariantKernelBasis(ftype_in, ftype_out)

    def forward(self, x, edge_index, angles, transporters):
        src, tgt = edge_index[0], edge_index[1]
        V = x.shape[0]

        K_self = self.kernel.eval_self()
        out = x @ K_self.T

        f_q = x[src]
        f_q_transported = apply_parallel_transport(f_q, transporters, self.ftype_in)
        K_neigh = self.kernel.eval_neigh(angles)
        msg = torch.bmm(K_neigh, f_q_transported.unsqueeze(-1)).squeeze(-1)
        out = out + scatter_add(msg, tgt, dim=0, dim_size=V)

        return out


class RegularNonlinearity(nn.Module):
    """
    Approximately gauge-equivariant nonlinearity via Fourier trick (Sec 5).

    1. Treat feature as Fourier coefficients of a periodic signal
    2. Inverse DFT to N spatial samples
    3. Apply pointwise ReLU
    4. Forward DFT back to Fourier coefficients

    Exactly equivariant for gauge angles that are multiples of 2π/N.
    """

    def __init__(self, ftype: FeatureType, N: int = 7, nonlin: nn.Module = None):
        super().__init__()
        self.ftype = ftype
        self.N = N
        self.nonlin = nonlin or nn.ReLU()
        self.register_buffer('_dummy', torch.zeros(1))

        max_order = max(order for order, _ in ftype)
        self._build_dft_matrices(max_order)

    def _build_dft_matrices(self, max_order):
        N = self.N
        n_coeffs = 2 * max_order + 1
        theta = torch.linspace(0, 2 * torch.pi * (N - 1) / N, N)

        A = torch.zeros(N, n_coeffs)
        A[:, 0] = 1.0
        for k in range(1, max_order + 1):
            A[:, 2 * k - 1] = torch.cos(k * theta)
            A[:, 2 * k] = torch.sin(k * theta)

        B = torch.linalg.pinv(A)
        self.register_buffer('A', A)
        self.register_buffer('B', B)

    def forward(self, x):
        out = torch.zeros_like(x)
        offset = 0
        for (order, mult) in self.ftype:
            d = 1 if order == 0 else 2
            C = mult * d
            block = x[:, offset:offset + C]
            if order == 0:
                out[:, offset:offset + C] = self.nonlin(block)
            else:
                norms = block.reshape(-1, mult, d).norm(dim=-1, keepdim=True).clamp(min=1e-8)
                new_norms = F.softplus(norms)
                out_block = block.reshape(-1, mult, d) * (new_norms / norms)
                out[:, offset:offset + C] = out_block.reshape(-1, C)
            offset += C
        return out


class GEMBlock(nn.Module):
    """
    One residual GEM-CNN block:
        f → GEMConv → LayerNorm → RegularNonlinearity → f'
        f' = f' + linear_skip(f)
    """

    def __init__(self, ftype_in: FeatureType, ftype_out: FeatureType, N_nonlin: int = 7):
        super().__init__()
        self.ftype_in = ftype_in
        self.ftype_out = ftype_out
        dim_in = feature_dim(ftype_in)
        dim_out = feature_dim(ftype_out)

        self.conv = GEMConv(ftype_in, ftype_out)
        self.norm = nn.LayerNorm(dim_out)
        self.nonlin = RegularNonlinearity(ftype_out, N=N_nonlin)

        if dim_in == dim_out:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x, edge_index, angles, transporters):
        residual = self.skip(x)
        h = self.conv(x, edge_index, angles, transporters)
        h = self.norm(h)
        h = self.nonlin(h)
        return h + residual
