"""
Mesh coarsening for the coarse-to-fine architecture.

Produces a decimated mesh and all mapping data needed for pool/unpool:
  - coarse_idx: which fine vertices survive
  - coarse_edge_index: edges of the coarsened mesh
  - coarse geometry: angles, transporters at coarse resolution
  - interp_matrix: sparse (V_fine, V_coarse) barycentric interpolation matrix
"""

import torch
import numpy as np
from typing import Dict, Optional

from data.mesh_geometry import (
    precompute_geometry, build_edge_index_from_faces,
)
from models.pooling import precompute_barycentric_weights


def decimate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_vertices: int,
) -> Dict:
    """
    Decimate a mesh to approximately target_vertices using quadric decimation.

    Falls back to random vertex sampling if pyvista is not available.

    Returns dict with:
      - coarse_verts: (V_coarse, 3)
      - coarse_faces: (F_coarse, 3)
      - coarse_idx: (V_coarse,) indices into original vertex array
    """
    V = vertices.shape[0]

    if V <= target_vertices:
        return {
            'coarse_verts': vertices,
            'coarse_faces': faces,
            'coarse_idx': np.arange(V),
        }

    try:
        import pyvista as pv

        flat_faces = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()
        mesh = pv.PolyData(vertices, flat_faces)

        target_reduction = 1.0 - (target_vertices / V)
        target_reduction = min(target_reduction, 0.97)

        decimated = mesh.decimate_pro(target_reduction)
        decimated = decimated.clean(tolerance=1e-6)

        coarse_verts = np.array(decimated.points, dtype=np.float32)
        coarse_faces_flat = np.array(decimated.faces)
        coarse_faces = coarse_faces_flat.reshape(-1, 4)[:, 1:].astype(np.int64)

        # Find closest original vertex for each coarse vertex
        from scipy.spatial import cKDTree
        tree = cKDTree(vertices)
        _, coarse_idx = tree.query(coarse_verts, k=1)

        return {
            'coarse_verts': coarse_verts,
            'coarse_faces': coarse_faces,
            'coarse_idx': coarse_idx,
        }

    except ImportError:
        # Fallback: random vertex subset with Delaunay triangulation
        coarse_idx = np.random.choice(V, target_vertices, replace=False)
        coarse_idx.sort()
        coarse_verts = vertices[coarse_idx]

        # Build simple triangulation from nearest neighbours
        from scipy.spatial import Delaunay
        try:
            tri = Delaunay(coarse_verts[:, :2])
            coarse_faces = tri.simplices.astype(np.int64)
        except Exception:
            coarse_faces = np.zeros((0, 3), dtype=np.int64)

        return {
            'coarse_verts': coarse_verts,
            'coarse_faces': coarse_faces,
            'coarse_idx': coarse_idx,
        }


def precompute_multires_data(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    target_coarse: int = 2000,
) -> Dict[str, torch.Tensor]:
    """
    Precompute all multi-resolution data for one mesh.

    Returns a dict containing everything the model needs:
      Fine mesh:  edge_index, angles, transporters, e1, e2
      Coarse mesh: coarse_idx, edge_index, angles, transporters
      Mapping: interp_matrix (sparse)
    """
    verts_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    device = vertices.device

    # 1. Fine mesh geometry
    fine_edge_index = build_edge_index_from_faces(faces)
    fine_geo = precompute_geometry(vertices, faces, fine_edge_index)

    # 2. Decimate
    dec = decimate_mesh(verts_np, faces_np, target_coarse)

    coarse_verts_t = torch.from_numpy(dec['coarse_verts']).to(device)
    coarse_faces_t = torch.from_numpy(dec['coarse_faces']).long().to(device)
    coarse_idx_t = torch.from_numpy(dec['coarse_idx']).long().to(device)

    # 3. Coarse mesh geometry
    if coarse_faces_t.shape[0] > 0:
        coarse_edge_index = build_edge_index_from_faces(coarse_faces_t)
        coarse_geo = precompute_geometry(coarse_verts_t, coarse_faces_t, coarse_edge_index)
    else:
        # Fallback: build edges from k-nearest neighbours
        from scipy.spatial import cKDTree
        tree = cKDTree(dec['coarse_verts'])
        k = min(6, len(dec['coarse_verts']) - 1)
        _, nn_idx = tree.query(dec['coarse_verts'], k=k + 1)
        edges = []
        for i in range(len(dec['coarse_verts'])):
            for j in nn_idx[i, 1:]:
                edges.append([i, j])
                edges.append([j, i])
        coarse_edge_index = torch.tensor(edges, dtype=torch.long, device=device).T
        coarse_edge_index = torch.unique(coarse_edge_index, dim=1)
        coarse_geo = precompute_geometry(coarse_verts_t, coarse_faces_t, coarse_edge_index)

    # 4. Barycentric interpolation matrix
    interp_matrix = precompute_barycentric_weights(
        verts_np, dec['coarse_verts'], dec['coarse_faces'], dec['coarse_idx']
    ).to(device)

    return {
        # Fine
        'fine_edge_index': fine_edge_index.to(device),
        'fine_angles': fine_geo['angles'],
        'fine_transporters': fine_geo['transporters'],
        'e1': fine_geo['e1'],
        'e2': fine_geo['e2'],
        # Coarse
        'coarse_idx': coarse_idx_t,
        'coarse_edge_index': coarse_edge_index.to(device),
        'coarse_angles': coarse_geo['angles'],
        'coarse_transporters': coarse_geo['transporters'],
        # Mapping
        'interp_matrix': interp_matrix,
    }
