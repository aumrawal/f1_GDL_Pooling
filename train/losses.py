"""
Combined loss for F1 aerodynamics: Cp + WSS + Cd + Cl.

Each component uses MSE with configurable weighting. The WSS loss
operates on 3D ambient vectors, which works for both the symmetry-breaking
MLP head and the equivariant GEM head (both produce ℝ³ output).
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple


class F1AeroLoss(nn.Module):
    """
    Weighted multi-task loss:
        L = w_cp * MSE(Cp) + w_wss * MSE(WSS) + w_cd * MSE(Cd) + w_cl * MSE(Cl)
    """

    def __init__(
        self,
        w_cp: float = 1.0,
        w_wss: float = 1.0,
        w_cd: float = 10.0,
        w_cl: float = 10.0,
    ):
        super().__init__()
        self.w_cp = w_cp
        self.w_wss = w_wss
        self.w_cd = w_cd
        self.w_cl = w_cl

    def forward(
        self, pred: Dict[str, Tensor], batch
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss_cp = nn.functional.mse_loss(pred['cp'], batch.y_cp)
        loss_wss = nn.functional.mse_loss(pred['wss'], batch.y_wss)
        loss_cd = nn.functional.mse_loss(pred['cd'], batch.y_cd)
        loss_cl = nn.functional.mse_loss(pred['cl'], batch.y_cl)

        total = (self.w_cp * loss_cp + self.w_wss * loss_wss
                 + self.w_cd * loss_cd + self.w_cl * loss_cl)

        parts = {'cp': loss_cp, 'wss': loss_wss, 'cd': loss_cd, 'cl': loss_cl}
        return total, parts
