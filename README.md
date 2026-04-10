# F1 Aero GEM-CNN V2: Coarse-to-Fine Architecture

Predicts CFD outputs (Cp, WSS, Cd, Cl) on F1 car meshes using a
coarse-to-fine Gauge Equivariant Mesh CNN — ~2.5× faster than the
flat architecture with improved global predictions.

## Architecture

```
Input [x,y,z,U] at 50k vertices
  │
  ├─ Linear embedding
  ├─ Pool to 2k vertices (precomputed decimation)
  ├─ 6× GEM blocks at 2k (fast, global receptive field)
  │     ├─ Cd/Cl heads (global mean-pool from 2k bottleneck)
  │
  ├─ Sym-break → barycentric interpolation → concat input → project
  ├─ 2× GEM blocks at 50k (refinement)
  │
  ├─ Cp head (symmetry-breaking MLP)
  └─ WSS head (equivariant GEMConv → ρ₁ → gauge map E_p)
```

## Key Design Decisions

- **Scalar heads (Cp, Cd, Cl)**: symmetry-breaking MLP bypasses Schur bottleneck
- **WSS head**: equivariant GEMConv ensures gauge-invariant 3D vector output
- **Interpolation**: only ρ₀ scalars are interpolated (avoids frame mismatch)
- **Refinement layers**: regenerate ρ₁ structure from geometry + interpolated scalars

## Install

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pyvista vtk scipy numpy pyyaml
```

## Project Structure

```
f1_aero_gem_v2/
├── models/
│   ├── irreps.py        # SO(2) irreps, kernel basis (Table 1)
│   ├── gem_conv.py      # GEMConv, RegularNonlinearity, GEMBlock
│   ├── pooling.py       # MeshPool, BarycentricInterpolator, MeshUnpool
│   ├── heads.py         # ScalarHead, GlobalHead, EquivariantWSSHead
│   └── f1_net_v2.py     # Main coarse-to-fine network
├── data/
│   ├── mesh_geometry.py # Tangent frames, log map, transporters
│   ├── coarsen.py       # Mesh decimation + multi-res precomputation
│   └── transforms.py    # Normalisation
├── train/
│   ├── losses.py        # Multi-task loss (Cp + WSS + Cd + Cl)
│   └── trainer.py       # Training loop
└── configs/
    └── f1_c2f.yaml      # Hyperparameters
```
