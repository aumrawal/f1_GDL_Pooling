"""
Core differential geometry operations for GEM-CNN on triangular meshes.

Implements (following de Haan et al. 2020, Secs 4.1 & 4.2):
  - Area-weighted vertex normals
  - Discrete Riemannian logarithmic map
  - Local reference frame construction (gauge choice)
  - Neighbour angle computation (θ_pq)
  - Discrete Levi-Civita parallel transporters (g_{q→p})
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict


def compute_vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """Area-weighted vertex normals. Returns (V, 3) unit normals."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = torch.linalg.cross(v1 - v0, v2 - v0)
    normals = torch.zeros_like(vertices)
    for i in range(3):
        idx = faces[:, i].unsqueeze(1).expand(-1, 3)
        normals.scatter_add_(0, idx, cross)
    return F.normalize(normals, dim=-1)


def log_map(p: Tensor, q: Tensor, normal_p: Tensor) -> Tensor:
    """Project edge (q-p) onto tangent plane at p, preserving edge length."""
    edge = q - p
    edge_len = edge.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dot = (edge * normal_p).sum(dim=-1, keepdim=True)
    proj = edge - dot * normal_p
    proj_len = proj.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return edge_len * proj / proj_len


def build_reference_frames(normals: Tensor, ref_vectors: Tensor) -> Tuple[Tensor, Tensor]:
    """Build (e1, e2) orthonormal frame from reference neighbour direction."""
    e1 = F.normalize(ref_vectors, dim=-1)
    e2 = torch.linalg.cross(normals, e1)
    e2 = F.normalize(e2, dim=-1)
    return e1, e2


def compute_neighbour_angles(log_pq: Tensor, e1_p: Tensor, e2_p: Tensor) -> Tensor:
    """Polar angle θ_pq of neighbour q in local frame at p."""
    cos_comp = (log_pq * e1_p).sum(dim=-1)
    sin_comp = (log_pq * e2_p).sum(dim=-1)
    return torch.atan2(sin_comp, cos_comp)


def compute_parallel_transporters(
    e1_src: Tensor, e2_src: Tensor, e1_tgt: Tensor,
    n_src: Tensor, n_tgt: Tensor,
) -> Tensor:
    """Discrete Levi-Civita connection g_{q→p} (Eq. 6 of GEM paper)."""
    cos_alpha = (n_src * n_tgt).sum(dim=-1).clamp(-1.0, 1.0)
    axis = torch.linalg.cross(n_src, n_tgt)
    axis_len = axis.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    axis = axis / axis_len
    sin_alpha = axis_len.squeeze(-1)

    def rodrigues_rotate(v, k, cos_a, sin_a):
        kdotv = (k * v).sum(dim=-1, keepdim=True)
        kcrossv = torch.linalg.cross(k, v)
        return v * cos_a.unsqueeze(-1) + kcrossv * sin_a.unsqueeze(-1) + k * kdotv * (1 - cos_a).unsqueeze(-1)

    e1_rot = rodrigues_rotate(e1_src, axis, cos_alpha, sin_alpha)
    e2_rot = rodrigues_rotate(e2_src, axis, cos_alpha, sin_alpha)

    cos_g = (e1_rot * e1_tgt).sum(dim=-1)
    sin_g = (e2_rot * e1_tgt).sum(dim=-1)
    return torch.atan2(sin_g, cos_g)


def build_edge_index_from_faces(faces: Tensor) -> Tensor:
    """Build bidirectional edge_index (2, E) from face tensor (F, 3)."""
    import numpy as np
    faces_np = faces.cpu().numpy()
    faces_sorted = np.sort(faces_np, axis=1)
    _, unique_idx = np.unique(faces_sorted, axis=0, return_index=True)
    faces_np = faces_np[unique_idx]
    faces = torch.from_numpy(faces_np).to(faces.device)

    pairs = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        e = torch.stack([faces[:, i], faces[:, j]], dim=0)
        e = torch.sort(e, dim=0)[0]
        pairs.append(e)
    edge_index = torch.cat(pairs, dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    # Make bidirectional
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def precompute_geometry(vertices: Tensor, faces: Tensor, edge_index: Tensor) -> Dict[str, Tensor]:
    """
    Precompute all geometric quantities needed by GEM-CNN.
    Returns dict with normals, e1, e2, angles, transporters.
    """
    device = vertices.device
    src, tgt = edge_index[0], edge_index[1]

    normals = compute_vertex_normals(vertices, faces)
    log_ij = log_map(p=vertices[tgt], q=vertices[src], normal_p=normals[tgt])

    # Reference neighbour: first neighbour of each vertex
    V = vertices.shape[0]
    full_edge_index = edge_index
    f_src, f_tgt = full_edge_index[0], full_edge_index[1]

    ref_edge = torch.full((V,), f_tgt.shape[0] - 1, dtype=torch.long, device=device)
    for i in range(f_tgt.shape[0] - 1, -1, -1):
        ref_edge[f_tgt[i]] = i
    ref_log = log_ij[ref_edge]

    e1, e2 = build_reference_frames(normals, ref_log)
    angles = compute_neighbour_angles(log_ij, e1[tgt], e2[tgt])
    transporters = compute_parallel_transporters(e1[src], e2[src], e1[tgt], normals[src], normals[tgt])

    return {
        'normals': normals, 'e1': e1, 'e2': e2,
        'angles': angles, 'transporters': transporters,
    }
