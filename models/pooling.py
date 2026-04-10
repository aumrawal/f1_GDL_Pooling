"""
Mesh pooling, unpooling, and barycentric interpolation for coarse-to-fine GEM-CNN.

This module implements the spatial resolution changes:
  - MeshPool: select a subset of vertices (precomputed or learned scoring)
  - BarycentricInterpolator: reconstruct fine features from coarse via
    precomputed barycentric coordinates
  - MeshUnpool: place coarse features back at original positions + interpolate

The key design decision: we interpolate only ρ₀ (scalar) features.
Vector (ρ₁) features cannot be naively interpolated because each vertex's
coefficients are expressed in a different tangent frame. Instead, we
break symmetry to all-scalar before unpooling, then let refinement GEM
layers regenerate ρ₁ structure from geometry.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
import numpy as np


class MeshPool(nn.Module):
    """
    Pool a mesh from V_fine vertices to V_coarse vertices.

    Uses precomputed vertex indices (from mesh decimation) rather than
    learned top-k scoring, since the coarse mesh is fixed across training.

    Stores:
      - coarse_idx: (V_coarse,) indices into the fine mesh
      - coarse_edge_index: (2, E_coarse) edges of the coarsened mesh
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: Tensor,              # (V_fine, C) features
        coarse_idx: Tensor,      # (V_coarse,) long
    ) -> Tensor:
        """Select features at coarse vertex positions."""
        return x[coarse_idx]     # (V_coarse, C)


class BarycentricInterpolator(nn.Module):
    """
    Interpolate features from V_coarse vertices to V_fine vertices using
    precomputed barycentric coordinates.

    For each fine vertex that is NOT in the coarse set, we find the triangle
    in the coarse mesh that contains it and compute barycentric weights
    (λ₁, λ₂, λ₃). The interpolated feature is:

        f_fine = λ₁ · f_a + λ₂ · f_b + λ₃ · f_c

    For fine vertices that ARE in the coarse set, we copy features directly.

    This is stored as a sparse matrix S of shape (V_fine, V_coarse) such that:
        f_fine = S @ f_coarse

    IMPORTANT: This interpolation is only correct for ρ₀ (scalar) features.
    For ρ₁ features, each coarse vertex's coefficients are in different frames,
    and averaging them without parallel transport is geometrically invalid.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_coarse: Tensor,              # (V_coarse, C)
        interp_matrix: Tensor,          # (V_fine, V_coarse) sparse or dense
    ) -> Tensor:
        """Interpolate coarse features to fine mesh."""
        if interp_matrix.is_sparse:
            return torch.sparse.mm(interp_matrix, x_coarse)
        return interp_matrix @ x_coarse  # (V_fine, C)


class MeshUnpool(nn.Module):
    """
    Reconstruct fine-mesh features from coarse-mesh features.

    Pipeline:
      1. Project coarse features to all-scalar (symmetry breaking)
      2. Interpolate to fine mesh via barycentric coordinates
      3. Concatenate with fine-mesh input features (skip connection)
      4. Linear projection to target dimension

    The symmetry breaking before interpolation avoids the ρ₁ frame
    mismatch problem entirely.
    """

    def __init__(
        self,
        coarse_dim: int,
        fine_input_dim: int,
        output_dim: int,
        scalar_proj_dim: int = 64,
    ):
        super().__init__()
        self.sym_break = nn.Linear(coarse_dim, scalar_proj_dim)
        self.interpolator = BarycentricInterpolator()
        self.fuse = nn.Linear(scalar_proj_dim + fine_input_dim, output_dim)

    def forward(
        self,
        x_coarse: Tensor,              # (V_coarse, coarse_dim)
        x_fine_input: Tensor,           # (V_fine, fine_input_dim)
        interp_matrix: Tensor,          # (V_fine, V_coarse)
    ) -> Tensor:
        # 1. Break symmetry: all features become scalars (gauge-invariant)
        x_scalar = self.sym_break(x_coarse)        # (V_coarse, scalar_proj_dim)

        # 2. Interpolate scalars to fine mesh
        x_interp = self.interpolator(x_scalar, interp_matrix)  # (V_fine, scalar_proj_dim)

        # 3. Concatenate with fine-mesh input (skip connection)
        x_cat = torch.cat([x_interp, x_fine_input], dim=-1)

        # 4. Project to output dimension
        return self.fuse(x_cat)                     # (V_fine, output_dim)


def precompute_barycentric_weights(
    fine_verts: np.ndarray,        # (V_fine, 3)
    coarse_verts: np.ndarray,      # (V_coarse, 3)
    coarse_faces: np.ndarray,      # (F_coarse, 3) indices into coarse_verts
    coarse_idx: np.ndarray,        # (V_coarse,) mapping: coarse → fine indices
) -> torch.Tensor:
    """
    Compute sparse interpolation matrix S of shape (V_fine, V_coarse).

    For each fine vertex:
      - If it's in the coarse set: S[i, coarse_pos] = 1.0
      - Otherwise: find nearest coarse triangle, compute barycentric coords

    Returns a sparse COO tensor.
    """
    from scipy.spatial import cKDTree

    V_fine = fine_verts.shape[0]
    V_coarse = coarse_verts.shape[0]

    # Build reverse mapping: fine_idx → coarse_idx
    fine_to_coarse = {}
    for ci, fi in enumerate(coarse_idx):
        fine_to_coarse[fi] = ci

    rows, cols, vals = [], [], []

    # Vertices in the coarse set: direct copy
    for fi, ci in fine_to_coarse.items():
        rows.append(fi)
        cols.append(ci)
        vals.append(1.0)

    # Vertices NOT in the coarse set: k-nearest-neighbour interpolation
    # (faster approximation of true barycentric — avoids triangle search)
    dropped = [i for i in range(V_fine) if i not in fine_to_coarse]

    if len(dropped) > 0 and V_coarse > 0:
        tree = cKDTree(coarse_verts)
        k = min(3, V_coarse)
        dists, nn_idx = tree.query(fine_verts[dropped], k=k)

        # Inverse-distance weighting
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists
        weights = weights / weights.sum(axis=1, keepdims=True)

        for i, fi in enumerate(dropped):
            for j in range(k):
                rows.append(fi)
                cols.append(nn_idx[i, j])
                vals.append(float(weights[i, j]))

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(V_fine, V_coarse))
