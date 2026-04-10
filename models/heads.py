"""
Output prediction heads for the coarse-to-fine F1 aero network.

Two fundamentally different head designs, motivated by Schur's lemma:

1. ScalarHead / GlobalHead: symmetry-breaking MLP
   - For Cp (per-vertex pressure), Cd (drag), Cl (downforce)
   - These are physical scalars (ρ₀) — gauge-invariant quantities
   - Uses nn.Linear which breaks symmetry (ignores irrep structure)
   - This lets all irrep channels (ρ₀, ρ₁, ρ₂) contribute to the
     scalar prediction, bypassing the Schur bottleneck

2. EquivariantWSSHead: GEMConv → ρ₁ → gauge map
   - WSS (wall shear stress) is physically a tangent vector (ρ₁)
   - Must transform correctly under gauge change
   - A GEMConv with output type 1ρ₁ produces equivariant (v₁, v₂)
   - The gauge map E_p converts to ambient ℝ³: WSS = v₁·e₁ + v₂·e₂
   - The output is gauge-invariant by construction
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_mean_pool
from typing import Optional

from models.gem_conv import GEMConv
from models.irreps import FeatureType, feature_dim


class ScalarHead(nn.Module):
    """
    Per-vertex scalar prediction (for Cp).

    Takes mixed-type features after symmetry breaking (treated as all ρ₀)
    and maps to a single scalar per vertex via MLP.

    The symmetry breaking happens upstream (in the main network), so this
    head receives plain ℝ^C features with no geometric structure.
    """

    def __init__(self, in_channels: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x).squeeze(-1)


class GlobalHead(nn.Module):
    """
    Global scalar prediction (for Cd, Cl).

    Mean-pools over all vertices, then MLP to single scalar.
    In the coarse-to-fine architecture, this can optionally operate
    at the coarse level (2k vertices) for more meaningful pooling.
    """

    def __init__(self, in_channels: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        pooled = global_mean_pool(x, batch)
        return self.mlp(pooled).squeeze(-1)


class EquivariantWSSHead(nn.Module):
    """
    Gauge-equivariant WSS prediction head using a GEMConv layer.

    This is the architecturally correct way to predict wall shear stress:

    1. GEMConv maps mixed features → 1ρ₁ (single tangent vector per vertex)
       - The kernel constraint ensures this map is equivariant
       - K_neigh can map from any ρₙ to ρ₁ (Table 1 of the paper)
       - This is NOT blocked by Schur's lemma (unlike K_self)

    2. The output (v₁, v₂) ∈ ℝ² are coefficients in the local tangent frame
       - Under gauge change g: (v₁, v₂) → ρ₁(-g) · (v₁, v₂)

    3. Gauge map converts to ambient 3D:
       WSS_3D = v₁ · e_{p,1} + v₂ · e_{p,2}
       - Different gauge → different (v₁, v₂) but same basis → same WSS_3D
       - This is gauge-invariant by construction

    The head also predicts a scalar magnitude correction via a separate
    ρ₀ → ρ₀ channel to improve the norm prediction.
    """

    def __init__(self, ftype_in: FeatureType):
        super().__init__()
        self.ftype_in = ftype_in

        # Equivariant conv: mixed type → 1ρ₁ (tangent vector)
        self.ftype_out = [(1, 1)]  # single ρ₁: 2D output
        self.gem_conv = GEMConv(ftype_in, self.ftype_out)

        # Optional: scalar magnitude predictor for norm correction
        self.ftype_mag = [(0, 1)]
        self.mag_conv = GEMConv(ftype_in, self.ftype_mag)

    def forward(
        self,
        x: Tensor,              # (V, C_in) mixed-type features
        edge_index: Tensor,      # (2, E)
        angles: Tensor,          # (E,)
        transporters: Tensor,    # (E,)
        e1: Tensor,              # (V, 3) first tangent basis vector
        e2: Tensor,              # (V, 3) second tangent basis vector
    ) -> Tensor:
        """
        Returns (V, 3) WSS vectors in ambient ℝ³.
        """
        # 1. Equivariant prediction: (V, 2) tangent coefficients
        tangent_coeffs = self.gem_conv(x, edge_index, angles, transporters)

        # 2. Scalar magnitude correction
        mag = self.mag_conv(x, edge_index, angles, transporters)  # (V, 1)
        mag_scale = torch.sigmoid(mag) * 2.0  # scale factor in [0, 2]

        # 3. Gauge map: 2D tangent → 3D ambient
        v1 = tangent_coeffs[:, 0:1]  # (V, 1)
        v2 = tangent_coeffs[:, 1:2]  # (V, 1)

        wss_3d = v1 * e1 + v2 * e2   # (V, 3) — gauge-invariant

        # 4. Apply magnitude correction
        wss_3d = wss_3d * mag_scale

        return wss_3d
