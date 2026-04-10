"""
SO(2) irreducible representations and gauge-equivariant kernel basis.

Implements the kernel constraint solution from Table 1 of
de Haan et al., "Gauge Equivariant Mesh CNNs", 2020.

Any gauge-equivariant kernel K_neigh(θ) mapping from irrep ρ_n to irrep ρ_m
must satisfy:
    K_neigh(θ - g) = ρ_m(-g) · K_neigh(θ) · ρ_n(g)   ∀ g,θ ∈ [0, 2π)

The solution space is spanned by basis kernels (fixed angular functions of θ)
multiplied by learned scalar weights.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

# A "feature type" is a list of (irrep_order, multiplicity) pairs.
# E.g. [(0, 16), (1, 16), (2, 8)] = 16ρ₀ ⊕ 16ρ₁ ⊕ 8ρ₂
FeatureType = List[Tuple[int, int]]


def feature_dim(ftype: FeatureType) -> int:
    """Total dimension of a feature type."""
    return sum(mult * (1 if order == 0 else 2) for order, mult in ftype)


def scalar_type(n_channels: int) -> FeatureType:
    """All-scalar feature type: n_channels × ρ₀."""
    return [(0, n_channels)]


def rho(order: int, angle: Tensor) -> Tensor:
    """
    SO(2) irrep ρ_n evaluated at angle g.

    ρ₀(g) = [[1]]
    ρ_n(g) = [[cos(ng), -sin(ng)],
              [sin(ng),  cos(ng)]]
    """
    if order == 0:
        return torch.ones(1, 1, dtype=angle.dtype, device=angle.device)
    g = order * angle
    c, s = torch.cos(g), torch.sin(g)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def rho_batch(order: int, angles: Tensor) -> Tensor:
    """Batched SO(2) irrep evaluation. Returns (B, d, d)."""
    if order == 0:
        return torch.ones(angles.shape[0], 1, 1, dtype=angles.dtype, device=angles.device)
    g = order * angles
    c, s = torch.cos(g), torch.sin(g)
    R = torch.zeros(angles.shape[0], 2, 2, dtype=angles.dtype, device=angles.device)
    R[:, 0, 0] = c;  R[:, 0, 1] = -s
    R[:, 1, 0] = s;  R[:, 1, 1] = c
    return R


def _neigh_basis_kernels(n_in: int, n_out: int, angles: Tensor) -> Tensor:
    """
    Evaluate basis kernels for K_neigh at given angles.
    Returns (E, d_out, d_in, n_basis).
    """
    E = angles.shape[0]

    if n_in == 0 and n_out == 0:
        return torch.ones(E, 1, 1, 1, dtype=angles.dtype, device=angles.device)

    if n_in == 0 and n_out > 0:
        m = n_out
        mt = m * angles
        cm, sm = torch.cos(mt), torch.sin(mt)
        B = torch.zeros(E, 2, 1, 2, dtype=angles.dtype, device=angles.device)
        B[:, 0, 0, 0] = cm;  B[:, 1, 0, 0] = sm
        B[:, 0, 0, 1] = sm;  B[:, 1, 0, 1] = -cm
        return B

    if n_in > 0 and n_out == 0:
        n = n_in
        nt = n * angles
        cn, sn = torch.cos(nt), torch.sin(nt)
        B = torch.zeros(E, 1, 2, 2, dtype=angles.dtype, device=angles.device)
        B[:, 0, 0, 0] = cn;  B[:, 0, 1, 0] = sn
        B[:, 0, 0, 1] = sn;  B[:, 0, 1, 1] = -cn
        return B

    n, m = n_in, n_out
    tp = (m + n) * angles
    tm = abs(m - n) * angles
    cp, sp = torch.cos(tp), torch.sin(tp)
    cm_, sm_ = torch.cos(tm), torch.sin(tm)

    B = torch.zeros(E, 2, 2, 4, dtype=angles.dtype, device=angles.device)
    B[:, 0, 0, 0] = cm_;  B[:, 0, 1, 0] = -sm_
    B[:, 1, 0, 0] = sm_;  B[:, 1, 1, 0] = cm_
    B[:, 0, 0, 1] = sm_;  B[:, 0, 1, 1] = cm_
    B[:, 1, 0, 1] = -cm_; B[:, 1, 1, 1] = sm_
    B[:, 0, 0, 2] = cp;   B[:, 0, 1, 2] = sp
    B[:, 1, 0, 2] = sp;   B[:, 1, 1, 2] = -cp
    B[:, 0, 0, 3] = -sp;  B[:, 0, 1, 3] = cp
    B[:, 1, 0, 3] = cp;   B[:, 1, 1, 3] = sp
    return B


def _self_basis_kernels(n_in: int, n_out: int) -> Tensor:
    """
    Basis kernels for K_self (Schur's lemma: only n_in == n_out allowed).
    Returns (d_out, d_in, n_basis) or None.
    """
    if n_in != n_out:
        return None
    if n_in == 0:
        return torch.ones(1, 1, 1)
    I = torch.eye(2).unsqueeze(-1)
    J = torch.tensor([[0., 1.], [-1., 0.]]).unsqueeze(-1)
    return torch.cat([I, J], dim=-1)


def _n_basis_neigh(n_in: int, n_out: int) -> int:
    if n_in == 0 and n_out == 0: return 1
    if n_in == 0 or n_out == 0: return 2
    return 4


class EquivariantKernelBasis(nn.Module):
    """
    Parameterised gauge-equivariant kernel K_neigh(θ) + K_self.

    Evaluates:
        K_neigh(θ) = Σ_i w_neigh_i * BasisKernel_i(θ)
        K_self     = Σ_i w_self_i  * BasisKernel_i
    """

    def __init__(self, ftype_in: FeatureType, ftype_out: FeatureType):
        super().__init__()
        self.ftype_in = ftype_in
        self.ftype_out = ftype_out
        self.dim_in = feature_dim(ftype_in)
        self.dim_out = feature_dim(ftype_out)

        n_neigh = sum(
            _n_basis_neigh(n_in, n_out) * m_in * m_out
            for (n_out, m_out) in ftype_out for (n_in, m_in) in ftype_in
        )
        n_self = sum(
            (1 if n_in == 0 else 2) * m_in * m_out
            for (n_out, m_out) in ftype_out for (n_in, m_in) in ftype_in
            if n_in == n_out
        )
        self.w_neigh = nn.Parameter(torch.randn(n_neigh) * 0.01)
        self.w_self = nn.Parameter(torch.randn(n_self) * 0.01)

    def eval_neigh(self, angles: Tensor) -> Tensor:
        """Returns (E, C_out, C_in) kernel matrix at each edge angle."""
        E = angles.shape[0]
        K = torch.zeros(E, self.dim_out, self.dim_in,
                        dtype=angles.dtype, device=angles.device)
        w_idx = 0
        o_off = 0
        for (n_out, m_out) in self.ftype_out:
            d_out = 1 if n_out == 0 else 2
            i_off = 0
            for (n_in, m_in) in self.ftype_in:
                d_in = 1 if n_in == 0 else 2
                B = _neigh_basis_kernels(n_in, n_out, angles)
                nb = B.shape[-1]
                for mo in range(m_out):
                    for mi in range(m_in):
                        w = self.w_neigh[w_idx:w_idx + nb]
                        w_idx += nb
                        block = (B * w.view(1, 1, 1, nb)).sum(-1)
                        ro = o_off + mo * d_out
                        ri = i_off + mi * d_in
                        K[:, ro:ro + d_out, ri:ri + d_in] = block
                i_off += m_in * d_in
            o_off += m_out * d_out
        return K

    def eval_self(self) -> Tensor:
        """Returns (C_out, C_in) self-interaction kernel."""
        K = torch.zeros(self.dim_out, self.dim_in,
                        dtype=self.w_self.dtype, device=self.w_self.device)
        w_idx = 0
        o_off = 0
        for (n_out, m_out) in self.ftype_out:
            d_out = 1 if n_out == 0 else 2
            i_off = 0
            for (n_in, m_in) in self.ftype_in:
                d_in = 1 if n_in == 0 else 2
                if n_in == n_out:
                    B = _self_basis_kernels(n_in, n_out).to(self.w_self.device)
                    nb = B.shape[-1]
                    for mo in range(m_out):
                        for mi in range(m_in):
                            w = self.w_self[w_idx:w_idx + nb]
                            w_idx += nb
                            block = (B * w.view(1, 1, nb)).sum(-1)
                            ro = o_off + mo * d_out
                            ri = i_off + mi * d_in
                            K[ro:ro + d_out, ri:ri + d_in] = block
                i_off += m_in * d_in
            o_off += m_out * d_out
        return K
