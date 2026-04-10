"""
Normalisation for F1 aero meshes.
Normalise vertex coordinates to [-1, 1]³ for consistent kernel learning.
"""

import torch
from torch import Tensor


def normalise_mesh(x: Tensor) -> Tensor:
    """Normalise XYZ to [-1, 1]³. U_inf channel (last column) unchanged."""
    xyz = x[:, :3]
    rest = x[:, 3:]
    xyz_min = xyz.min(dim=0).values
    xyz_max = xyz.max(dim=0).values
    centre = (xyz_min + xyz_max) / 2.0
    scale = (xyz_max - xyz_min).max().clamp(min=1e-8) / 2.0
    xyz_norm = (xyz - centre) / scale
    return torch.cat([xyz_norm, rest], dim=-1)
