"""
DrivAerNet dataset loader for the coarse-to-fine (V2) architecture.

Reads merged VTP files (produced by prepare_data.py from the V1 project)
and computes all multi-resolution data needed by F1AeroNetV2:
  - Fine mesh geometry (edges, angles, transporters, tangent frames)
  - Coarse mesh via decimation (2k vertices)
  - Coarse mesh geometry
  - Barycentric interpolation matrix

All results are cached as .pt files for fast reload.

Usage:
    from data.drivaernet_dataset_v2 import DrivAerNetV2Dataset
    from torch_geometric.loader import DataLoader

    train_ds = DrivAerNetV2Dataset(data_root='path/to/drivaernet_real', split='train')
    loader = DataLoader(train_ds, batch_size=2, shuffle=True)
"""

import os
import json
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from scipy.spatial import cKDTree

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from data.mesh_geometry import precompute_geometry, build_edge_index_from_faces
from data.transforms import normalise_mesh
from data.coarsen import decimate_mesh
from models.pooling import precompute_barycentric_weights


def load_vtp_to_v2_data(
    vtp_path: str,
    target_coarse: int = 2000,
    rho: float = 1.225,
    U_inf: float = 83.33,
    design_id: str = "",
) -> Data:
    """
    Load a single VTP file and compute all multi-resolution data
    needed by F1AeroNetV2.

    Steps:
      1. Read VTP: vertices, faces, pressure, WSS, Cd, Cl
      2. Build input features [x, y, z, U_inf] normalised to [-1,1]³
      3. Compute fine mesh geometry (edges, angles, transporters, e1, e2)
      4. Decimate to target_coarse vertices
      5. Compute coarse mesh geometry
      6. Build barycentric interpolation matrix (V_fine, V_coarse)
      7. Package everything into a PyG Data object

    Returns:
        PyG Data with attributes:
          x, fine_edge_index, fine_angles, fine_transporters, e1, e2,
          coarse_idx, coarse_edge_index, coarse_angles, coarse_transporters,
          interp_matrix, y_cp, y_wss, y_cd, y_cl
    """
    if not HAS_PYVISTA:
        raise RuntimeError("pip install pyvista vtk")

    # ── 1. Read VTP ───────────────────────────────────────────────
    mesh = pv.read(vtp_path).triangulate().clean(tolerance=1e-6)
    vertices = np.array(mesh.points, dtype=np.float32)
    faces = np.array(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int64)

    pressure = None
    for name in ['p', 'pressure', 'Pressure', 'pMean']:
        if name in mesh.point_data:
            arr = np.array(mesh.point_data[name], dtype=np.float32)
            pressure = arr[:, 0] if arr.ndim > 1 else arr
            break

    wss = None
    for name in ['wallShearStress', 'WallShearStress', 'wss', 'WSS']:
        if name in mesh.point_data:
            wss = np.array(mesh.point_data[name], dtype=np.float32).reshape(-1, 3)
            break

    cd_val = float(mesh.field_data['cd'][0]) if 'cd' in mesh.field_data else 0.0
    cl_val = float(mesh.field_data['cl'][0]) if 'cl' in mesh.field_data else 0.0

    # ── 2. Input features ─────────────────────────────────────────
    verts_t = torch.from_numpy(vertices)
    faces_t = torch.from_numpy(faces).long()
    U_col = torch.full((verts_t.shape[0], 1), U_inf)
    x = normalise_mesh(torch.cat([verts_t, U_col], dim=-1))

    q_inf = 0.5 * rho * U_inf ** 2
    if pressure is not None:
        p_ref = float(np.mean(pressure))
        y_cp = torch.from_numpy((pressure - p_ref) / q_inf).float()
    else:
        y_cp = torch.zeros(verts_t.shape[0])

    y_wss = torch.from_numpy(wss / q_inf).float() if wss is not None \
            else torch.zeros(verts_t.shape[0], 3)
    y_cd = torch.tensor([cd_val])
    y_cl = torch.tensor([cl_val])

    # ── 3. Fine mesh geometry ─────────────────────────────────────
    fine_edge_index = build_edge_index_from_faces(faces_t)
    fine_geo = precompute_geometry(verts_t, faces_t, fine_edge_index)

    # ── 4. Decimate to coarse mesh ────────────────────────────────
    dec = decimate_mesh(vertices, faces, target_coarse)
    coarse_verts_t = torch.from_numpy(dec['coarse_verts'])
    coarse_faces_t = torch.from_numpy(dec['coarse_faces']).long()
    coarse_idx_t = torch.from_numpy(dec['coarse_idx']).long()

    # ── 5. Coarse mesh geometry ───────────────────────────────────
    if coarse_faces_t.shape[0] > 0:
        coarse_edge_index = build_edge_index_from_faces(coarse_faces_t)
        coarse_geo = precompute_geometry(
            coarse_verts_t, coarse_faces_t, coarse_edge_index
        )
    else:
        tree = cKDTree(dec['coarse_verts'])
        k = min(6, len(dec['coarse_verts']) - 1)
        _, nn_idx = tree.query(dec['coarse_verts'], k=k + 1)
        edges = []
        for i in range(len(dec['coarse_verts'])):
            for j in nn_idx[i, 1:]:
                edges.extend([[i, j], [j, i]])
        coarse_edge_index = torch.unique(
            torch.tensor(edges, dtype=torch.long).T, dim=1
        )
        coarse_geo = precompute_geometry(
            coarse_verts_t, coarse_faces_t, coarse_edge_index
        )

    # ── 6. Interpolation matrix ───────────────────────────────────
    interp_matrix = precompute_barycentric_weights(
        vertices, dec['coarse_verts'], dec['coarse_faces'], dec['coarse_idx']
    )

    # ── 7. Assemble PyG Data ──────────────────────────────────────
    return Data(
        x=x,
        # Fine
        fine_edge_index=fine_edge_index,
        fine_angles=fine_geo['angles'],
        fine_transporters=fine_geo['transporters'],
        e1=fine_geo['e1'],
        e2=fine_geo['e2'],
        # Coarse
        coarse_idx=coarse_idx_t,
        coarse_edge_index=coarse_edge_index,
        coarse_angles=coarse_geo['angles'],
        coarse_transporters=coarse_geo['transporters'],
        # Mapping (dense for PyG batching compatibility)
        interp_matrix=interp_matrix.to_dense(),
        # Targets
        y_cp=y_cp, y_wss=y_wss, y_cd=y_cd, y_cl=y_cl,
        # Meta
        num_nodes=verts_t.shape[0],
        design_id=design_id,
    )


class DrivAerNetV2Dataset(Dataset):
    """
    Dataset that reads V1 VTP files and produces V2 multi-resolution Data.
    Caches processed .pt files for fast reload.

    Args:
        data_root: path to drivaernet_real/ (must contain meshes/ and split.json)
        split: 'train', 'val', or 'test'
        target_coarse: number of vertices in the coarse mesh
        rho: air density (kg/m³)
        U_inf: freestream velocity (m/s)
        cache_dir: where to write .pt cache (default: data_root/processed_v2/split)
        force_reload: if True, recompute even if cache exists
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        target_coarse: int = 2000,
        rho: float = 1.225,
        U_inf: float = 83.33,
        cache_dir: str = None,
        force_reload: bool = False,
    ):
        self.data_root = data_root
        self.split = split
        self.target_coarse = target_coarse
        self.rho = rho
        self.U_inf = U_inf
        self.force_reload = force_reload

        # Read split
        split_file = os.path.join(data_root, 'split.json')
        if os.path.exists(split_file):
            with open(split_file) as f:
                self.design_ids = json.load(f)[split]
        else:
            all_vtps = sorted([
                os.path.splitext(f)[0]
                for f in os.listdir(os.path.join(data_root, 'meshes'))
                if f.endswith('.vtp')
            ])
            n = len(all_vtps)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            if split == 'train':
                self.design_ids = all_vtps[:n_train]
            elif split == 'val':
                self.design_ids = all_vtps[n_train:n_train + n_val]
            else:
                self.design_ids = all_vtps[n_train + n_val:]

        # Cache directory
        if cache_dir is not None:
            self.cache_dir = os.path.join(cache_dir, split)
        else:
            self.cache_dir = os.path.join(data_root, 'processed_v2', split)
        os.makedirs(self.cache_dir, exist_ok=True)

        super().__init__(root=None)

    def len(self):
        return len(self.design_ids)

    def get(self, idx):
        did = self.design_ids[idx]
        cache_path = os.path.join(self.cache_dir, f'{did}.pt')

        if os.path.exists(cache_path) and not self.force_reload:
            return torch.load(cache_path, weights_only=False)

        vtp_path = os.path.join(self.data_root, 'meshes', f'{did}.vtp')
        if not os.path.exists(vtp_path):
            raise FileNotFoundError(f"Missing: {vtp_path}")

        print(f'  Processing {did} ({idx + 1}/{len(self.design_ids)})...')
        data = load_vtp_to_v2_data(
            vtp_path,
            target_coarse=self.target_coarse,
            rho=self.rho,
            U_inf=self.U_inf,
            design_id=did,
        )
        torch.save(data, cache_path)
        return data
