"""
Training loop for F1AeroNetV2 (coarse-to-fine architecture).

Usage:
    python -m train.trainer --config configs/f1_c2f.yaml

The full pipeline: load config → load data → build model → train.
All multi-resolution preprocessing (decimation, geometry, interpolation
matrices) is handled by DrivAerNetV2Dataset and cached to disk.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import yaml
import os
import time

from models.f1_net_v2 import F1AeroNetV2
from train.losses import F1AeroLoss
from data.drivaernet_dataset_v2 import DrivAerNetV2Dataset


def load_datasets(cfg: dict):
    """
    Load train/val datasets from drivaernet_real VTP files.
    Returns (train_loader, val_loader).
    """
    data_cfg = cfg['data']
    data_root = data_cfg['data_root']
    batch_size = cfg['training']['batch_size']

    train_ds = DrivAerNetV2Dataset(
        data_root=data_root,
        split='train',
        target_coarse=data_cfg.get('target_coarse', 2000),
        rho=data_cfg.get('rho', 1.225),
        U_inf=data_cfg.get('U_inf', 83.33),
        cache_dir=data_cfg.get('cache_dir', None),
    )
    val_ds = DrivAerNetV2Dataset(
        data_root=data_root,
        split='val',
        target_coarse=data_cfg.get('target_coarse', 2000),
        rho=data_cfg.get('rho', 1.225),
        U_inf=data_cfg.get('U_inf', 83.33),
        cache_dir=data_cfg.get('cache_dir', None),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} meshes ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds)} meshes ({len(val_loader)} batches)")
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, criterion, device,
                grad_clip=1.0, accum_steps=4):
    """
    Run one training epoch with gradient accumulation.

    With batch_size=2 in the DataLoader and accum_steps=4, the effective
    batch size is 2 × 4 = 8 meshes per optimizer step. This gives:
      - 400k supervised vertices per gradient update (plenty of signal)
      - 8 Cd/Cl labels per update (reasonable for global heads)
      - Same peak memory as batch_size=2 (~6.5 GB on P100)

    The loss is scaled by 1/accum_steps before .backward() so that the
    accumulated gradient magnitude matches what a true batch_size=8
    would produce.
    """
    model.train()
    total_loss = 0.0
    n = 0

    optimizer.zero_grad()  # zero once at the start

    for i, batch in enumerate(loader):
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

        loss, parts = criterion(pred, batch)
        if torch.isnan(loss):
            continue

        # Scale loss so accumulated gradients match true large-batch gradients
        scaled_loss = loss / accum_steps
        scaled_loss.backward()

        # Step optimizer every accum_steps batches (or at end of epoch)
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()  # track unscaled loss for logging
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
    """
    Full training pipeline: config → data → model → train loop → checkpoint.

    Usage:
        python -m train.trainer --config configs/f1_c2f.yaml
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Device selection: CUDA (Kaggle/Colab) > MPS (Mac M4) > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Device: {device} (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"Device: {device}")

    # Load data
    train_loader, val_loader = load_datasets(cfg)

    # Build model
    model = F1AeroNetV2.from_config(cfg).to(device)
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,}")

    # Optimizer
    accum_steps = cfg['training'].get('accum_steps', 4)
    batch_size = cfg['training']['batch_size']
    print(f"Batch size: {batch_size} x {accum_steps} accumulation = "
          f"{batch_size * accum_steps} effective")

    optimizer = Adam(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training'].get('weight_decay', 1e-5),
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = F1AeroLoss(**cfg.get('loss', {}))

    run_dir = cfg['training'].get('run_dir', 'runs')
    os.makedirs(run_dir, exist_ok=True)

    # Train
    epochs = cfg['training'].get('epochs', 50)
    grad_clip = cfg['training'].get('grad_clip', 1.0)
    best_val = float('inf')

    print(f"\nStarting training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        t_loss = train_epoch(model, train_loader, optimizer, criterion,
                             device, grad_clip=grad_clip, accum_steps=accum_steps)
        v_loss = validate(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        marker = ''
        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_v2.pt'))
            marker = ' *'

        print(f"Epoch {epoch:3d}/{epochs}  train={t_loss:.5f}  val={v_loss:.5f}  "
              f"lr={lr_now:.1e}  {elapsed:.0f}s{marker}")

    torch.save(model.state_dict(), os.path.join(run_dir, 'final_v2.pt'))
    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Models saved to {run_dir}/")

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/f1_c2f.yaml')
    args = parser.parse_args()
    train(args.config)