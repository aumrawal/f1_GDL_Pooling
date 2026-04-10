"""
F1AeroNet V2: Coarse-to-fine architecture for F1 aerodynamics.

Architecture:
    Input: per-vertex [x, y, z, U_inf] at V_fine vertices (e.g. 50k)
      │
      ├─ Input embedding: Linear(4, C₀) at fine mesh
      │
      ├─ Pool: select V_coarse vertices (e.g. 2k) via precomputed decimation
      │
      ├─ N × GEMBlock at coarse resolution (fast, large receptive field)
      │     Each block: GEMConv → LayerNorm → RegularNonlinearity → Residual
      │     Uses precomputed coarse geometry (angles, transporters)
      │
      ├─ Branch A: Global heads (Cd, Cl) from coarse features
      │     Mean-pool over 2k vertices → MLP (better than pooling 50k)
      │
      ├─ Unpool: sym_break → barycentric interpolation → concat skip → project
      │     Scalar interpolation avoids ρ₁ frame mismatch problem
      │
      ├─ M × GEMBlock at fine resolution (refine local detail)
      │     Uses precomputed fine geometry
      │
      ├─ Branch B: Cp head (symmetry-breaking MLP)
      │     Treats all features as scalars for maximum expressiveness
      │
      └─ Branch C: WSS head (equivariant GEMConv → ρ₁ → gauge map)
            Produces gauge-invariant 3D vectors via tangent frame decode

Cost analysis (50k fine, 2k coarse, 6 coarse + 2 refine blocks):
  - Coarse blocks: 6 × 0.45 GF = 2.7 GF (vs 68 GF current)
  - Refine blocks: 2 × 11.3 GF = 22.6 GF
  - Total: ~27 GF (vs ~69 GF current) → ~2.5× faster
  - Peak memory: ~2-3 GB (vs ~5-6 GB current)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional, Dict

from models.gem_conv import GEMBlock
from models.irreps import FeatureType, feature_dim
from models.pooling import MeshPool, MeshUnpool
from models.heads import ScalarHead, GlobalHead, EquivariantWSSHead


def build_ftype(mult: int, max_order: int) -> FeatureType:
    """Build feature type with mult copies of each irrep up to max_order."""
    return [(order, mult) for order in range(max_order + 1)]


class F1AeroNetV2(nn.Module):
    """
    Coarse-to-fine GEM-CNN for F1 car aerodynamics.

    Args:
        in_channels: per-vertex input features (default 4: x, y, z, U_inf)
        coarse_specs: list of (mult, max_order) for coarse GEM blocks
        refine_specs: list of (mult, max_order) for fine refinement blocks
        N_nonlin: RegularNonlinearity sample count
        scalar_proj_dim: dimension after symmetry breaking for interpolation
        head_hidden: hidden dim in prediction MLP heads
        head_dropout: dropout in prediction heads
    """

    def __init__(
        self,
        in_channels: int = 4,
        coarse_specs: List[Tuple[int, int]] = None,
        refine_specs: List[Tuple[int, int]] = None,
        N_nonlin: int = 7,
        scalar_proj_dim: int = 64,
        head_hidden: int = 128,
        head_dropout: float = 0.1,
    ):
        super().__init__()

        if coarse_specs is None:
            coarse_specs = [(16, 2), (32, 2), (32, 3), (64, 3), (64, 2), (64, 1)]
        if refine_specs is None:
            refine_specs = [(32, 1), (32, 1)]

        # ── Input embedding at fine resolution ────────────────────────
        first_coarse_ftype = build_ftype(coarse_specs[0][0], coarse_specs[0][1])
        first_coarse_dim = feature_dim(first_coarse_ftype)
        self.input_embed = nn.Linear(in_channels, first_coarse_dim)

        # ── Pooling ───────────────────────────────────────────────────
        self.pool = MeshPool()

        # ── Coarse GEM blocks ─────────────────────────────────────────
        self.coarse_blocks = nn.ModuleList()
        self.coarse_ftypes = [first_coarse_ftype]

        for mult, max_order in coarse_specs:
            ftype_out = build_ftype(mult, max_order)
            self.coarse_blocks.append(
                GEMBlock(self.coarse_ftypes[-1], ftype_out, N_nonlin)
            )
            self.coarse_ftypes.append(ftype_out)

        last_coarse_ftype = self.coarse_ftypes[-1]
        last_coarse_dim = feature_dim(last_coarse_ftype)

        # ── Global heads (Cd, Cl) from coarse bottleneck ──────────────
        self.coarse_sym_break = nn.Linear(last_coarse_dim, scalar_proj_dim)
        self.cd_head = GlobalHead(scalar_proj_dim, hidden=head_hidden // 2,
                                 dropout=head_dropout)
        self.cl_head = GlobalHead(scalar_proj_dim, hidden=head_hidden // 2,
                                 dropout=head_dropout)

        # ── Unpooling: coarse → fine ──────────────────────────────────
        self.unpool = MeshUnpool(
            coarse_dim=last_coarse_dim,
            fine_input_dim=in_channels,
            output_dim=feature_dim(build_ftype(refine_specs[0][0], refine_specs[0][1])),
            scalar_proj_dim=scalar_proj_dim,
        )

        # ── Fine refinement GEM blocks ────────────────────────────────
        self.refine_blocks = nn.ModuleList()
        first_refine_ftype = build_ftype(refine_specs[0][0], refine_specs[0][1])
        self.refine_ftypes = [first_refine_ftype]

        for i, (mult, max_order) in enumerate(refine_specs):
            ftype_out = build_ftype(mult, max_order)
            ftype_in = self.refine_ftypes[-1]
            self.refine_blocks.append(
                GEMBlock(ftype_in, ftype_out, N_nonlin)
            )
            self.refine_ftypes.append(ftype_out)

        last_refine_ftype = self.refine_ftypes[-1]
        last_refine_dim = feature_dim(last_refine_ftype)

        # ── Cp head (symmetry-breaking scalar) ────────────────────────
        self.cp_sym_break = nn.Linear(last_refine_dim, scalar_proj_dim)
        self.cp_head = ScalarHead(scalar_proj_dim, hidden=head_hidden,
                                 dropout=head_dropout)

        # ── WSS head (equivariant GEM) ────────────────────────────────
        self.wss_head = EquivariantWSSHead(last_refine_ftype)

    def forward(
        self,
        x: Tensor,                          # (V_fine, 4)
        fine_edge_index: Tensor,             # (2, E_fine)
        fine_angles: Tensor,                 # (E_fine,)
        fine_transporters: Tensor,           # (E_fine,)
        coarse_idx: Tensor,                  # (V_coarse,)
        coarse_edge_index: Tensor,           # (2, E_coarse)
        coarse_angles: Tensor,              # (E_coarse,)
        coarse_transporters: Tensor,         # (E_coarse,)
        interp_matrix: Tensor,               # (V_fine, V_coarse) sparse
        e1: Tensor,                          # (V_fine, 3) tangent basis
        e2: Tensor,                          # (V_fine, 3) tangent basis
        batch: Optional[Tensor] = None,
        coarse_batch: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the coarse-to-fine architecture.

        Returns dict with keys 'cp', 'wss', 'cd', 'cl'.
        """
        # ── 1. Input embedding at fine resolution ──────────────────────
        h_fine_input = self.input_embed(x)           # (V_fine, C₀)

        # ── 2. Pool to coarse mesh ─────────────────────────────────────
        h_coarse = self.pool(h_fine_input, coarse_idx)  # (V_coarse, C₀)

        # ── 3. Coarse GEM processing ──────────────────────────────────
        for block in self.coarse_blocks:
            h_coarse = block(h_coarse, coarse_edge_index,
                           coarse_angles, coarse_transporters)

        # ── 4. Global predictions from coarse bottleneck ───────────────
        h_coarse_scalar = self.coarse_sym_break(h_coarse)
        cd = self.cd_head(h_coarse_scalar, coarse_batch)
        cl = self.cl_head(h_coarse_scalar, coarse_batch)

        # ── 5. Unpool: interpolate coarse features to fine mesh ────────
        h_fine = self.unpool(h_coarse, x, interp_matrix)

        # ── 6. Fine refinement ─────────────────────────────────────────
        for block in self.refine_blocks:
            h_fine = block(h_fine, fine_edge_index,
                         fine_angles, fine_transporters)

        # ── 7. Cp prediction (symmetry-breaking scalar head) ───────────
        h_cp = self.cp_sym_break(h_fine)
        cp = self.cp_head(h_cp)

        # ── 8. WSS prediction (equivariant GEM head) ──────────────────
        wss = self.wss_head(
            h_fine, fine_edge_index, fine_angles, fine_transporters, e1, e2
        )

        return {'cp': cp, 'wss': wss, 'cd': cd, 'cl': cl}

    def count_parameters(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            'input_embed': count(self.input_embed),
            'coarse_blocks': sum(count(b) for b in self.coarse_blocks),
            'refine_blocks': sum(count(b) for b in self.refine_blocks),
            'unpool': count(self.unpool),
            'cp_head': count(self.cp_head) + count(self.cp_sym_break),
            'wss_head': count(self.wss_head),
            'cd_head': count(self.cd_head) + count(self.coarse_sym_break),
            'cl_head': count(self.cl_head),
            'total': count(self),
        }

    @classmethod
    def from_config(cls, cfg: dict) -> 'F1AeroNetV2':
        model_cfg = cfg.get('model', cfg)
        coarse_specs = [tuple(s) for s in model_cfg.get(
            'coarse_specs', [(16, 2), (32, 2), (32, 3), (64, 3), (64, 2), (64, 1)]
        )]
        refine_specs = [tuple(s) for s in model_cfg.get(
            'refine_specs', [(32, 1), (32, 1)]
        )]
        return cls(
            in_channels=model_cfg.get('in_channels', 4),
            coarse_specs=coarse_specs,
            refine_specs=refine_specs,
            N_nonlin=model_cfg.get('N_nonlin', 7),
            scalar_proj_dim=model_cfg.get('scalar_proj_dim', 64),
            head_hidden=model_cfg.get('head_hidden', 128),
            head_dropout=model_cfg.get('head_dropout', 0.1),
        )
