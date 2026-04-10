"""
Training loop for F1AeroNetV2 (coarse-to-fine architecture).

Key difference from V1: the forward pass requires both fine and coarse
mesh data (edges, angles, transporters, interpolation matrix), all of
which are precomputed and stored as graph attributes.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import os

from models.f1_net_v2 import F1AeroNetV2
from train.losses import F1AeroLoss


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(
            x=batch.x,
            fine_edge_index=batch.fine_edge_index,
            fine_angles=batch.fine_angles,
            fine_transporters=batch.fine_transporters,
            coarse_idx=batch.coarse_idx,
            coarse_edge_index=batch.coarse_edge_index,
            coarse_angles=batch.coarse_angles,
            coarse_transporters=batch.coarse_transporters,
            interp_matrix=batch.interp_matrix,
            e1=batch.e1,
            e2=batch.e2,
            batch=getattr(batch, 'batch', None),
            coarse_batch=getattr(batch, 'coarse_batch', None),
        )

        loss, parts = criterion(pred, batch)
        if torch.isnan(loss):
            continue

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(
            x=batch.x,
            fine_edge_index=batch.fine_edge_index,
            fine_angles=batch.fine_angles,
            fine_transporters=batch.fine_transporters,
            coarse_idx=batch.coarse_idx,
            coarse_edge_index=batch.coarse_edge_index,
            coarse_angles=batch.coarse_angles,
            coarse_transporters=batch.coarse_transporters,
            interp_matrix=batch.interp_matrix,
            e1=batch.e1,
            e2=batch.e2,
            batch=getattr(batch, 'batch', None),
            coarse_batch=getattr(batch, 'coarse_batch', None),
        )
        loss, _ = criterion(pred, batch)
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


def train(cfg_path: str):
    """Main training entry point."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    model = F1AeroNetV2.from_config(cfg).to(device)
    print(f"Parameters: {model.count_parameters()}")

    optimizer = Adam(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training'].get('weight_decay', 1e-5),
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = F1AeroLoss(**cfg.get('loss', {}))

    os.makedirs(cfg['training'].get('run_dir', 'runs'), exist_ok=True)

    print("Training loop ready. Provide DataLoaders to begin.")
    return model, optimizer, scheduler, criterion


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/f1_c2f.yaml')
    args = parser.parse_args()
    train(args.config)
