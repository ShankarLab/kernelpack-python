from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import distance_matrix
from kernelpack.poly import total_degree_indices
from kernelpack.rbffd import OpProperties, RBFStencil, StencilProperties


@dataclass
class PatchCenterTree:
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    searcher: cKDTree | None = None
    has_searcher: bool = False


@dataclass
class PUPatchData:
    centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    radius: float = 0.0
    spacing: float = 0.0
    min_patch_nodes: int = 0
    node_ids: list[np.ndarray] = field(default_factory=list)
    center_tree: PatchCenterTree = field(default_factory=PatchCenterTree)
    stencil_props: StencilProperties = field(default_factory=StencilProperties)
    stencils: list[RBFStencil] = field(default_factory=list)


def pu_patch_geometry(
    domain: DomainDescriptor,
    xi: int,
    patch_spacing_factor: float = 0.0,
    patch_radius_factor: float = 0.0,
) -> PUPatchData:
    domain.build_structs()
    xf = domain.get_all_nodes()
    h = domain.get_sep_rad()
    spacing = choose_patch_spacing(h, patch_spacing_factor)
    radius = choose_patch_radius(h, patch_radius_factor)
    min_nodes = choose_minimum_patch_nodes(xf.shape[1], xi)
    centers = choose_patch_centers(xf, spacing)
    node_ids = build_patch_node_ids(domain, centers, radius, min_nodes, "all")
    stencil_props = build_patch_stencil_properties(xf.shape[1], xi)
    stencils = build_patch_stencils(xf, node_ids, stencil_props)
    return PUPatchData(
        centers=centers,
        radius=radius,
        spacing=spacing,
        min_patch_nodes=min_nodes,
        node_ids=node_ids,
        center_tree=build_center_tree(centers),
        stencil_props=stencil_props,
        stencils=stencils,
    )


def pu_localized_operator(
    domain: DomainDescriptor,
    patch_data: PUPatchData,
    xi: int,
    xq: np.ndarray,
    op_name: str,
    *,
    normals: np.ndarray | None = None,
    neu_coeff: np.ndarray | None = None,
    dir_coeff: np.ndarray | None = None,
) -> sparse.csr_matrix:
    xq = np.asarray(xq, dtype=float)
    if xq.size == 0:
        return sparse.csr_matrix((0, domain.get_num_total_nodes()))

    xnodes = domain.get_all_nodes()
    centers = patch_data.centers
    node_ids = patch_data.node_ids
    cached_stencils = patch_data.stencils
    radius = patch_data.radius
    dim = xnodes.shape[1]
    num_targets = xq.shape[0]
    num_all = xnodes.shape[0]
    patch_ids_per_query = query_patch_ids(patch_data, xq, radius)

    normals = np.zeros((num_targets, dim)) if normals is None else np.asarray(normals, dtype=float)
    neu_coeff = np.zeros(num_targets) if neu_coeff is None else np.asarray(neu_coeff, dtype=float).reshape(-1)
    dir_coeff = np.zeros(num_targets) if dir_coeff is None else np.asarray(dir_coeff, dtype=float).reshape(-1)

    sp = copy_stencil_properties(patch_data.stencil_props)
    name = str(op_name).lower()
    if name in {"interp", "interpolation"}:
        theta = 0
    elif name in {"lap", "laplacian"}:
        theta = 2
    elif name in {"bc", "boundary"}:
        theta = 1
    else:
        raise ValueError(f'unknown PU operator "{op_name}"')
    if theta != 1:
        sp.ell = max(xi + theta - 1, 2)
        sp.npoly = total_degree_indices(dim, sp.ell).shape[0]
        sp.spline_degree = max(5, sp.ell)
        if sp.spline_degree % 2 == 0:
            sp.spline_degree -= 1

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for q in range(num_targets):
        patch_ids = patch_ids_per_query[q]
        center_dist = np.linalg.norm(centers - xq[q], axis=1)
        alpha = pu_patch_weight(center_dist[patch_ids] / radius)
        alpha_sum = float(alpha.sum())
        if alpha_sum <= 1.0e-14:
            alpha = np.ones_like(alpha)
            alpha_sum = float(alpha.sum())
        alpha = alpha / alpha_sum

        row = np.zeros(num_all)
        for k, p in enumerate(patch_ids):
            ids = node_ids[p]
            stencil = cached_stencils[p] if theta == 2 else build_patch_stencil(xnodes[ids], sp)
            wloc = local_operator_weights(stencil, xq[q], sp, op_name, normals[q], float(neu_coeff[q]), float(dir_coeff[q]))
            row[ids] += alpha[k] * wloc

        nz = np.flatnonzero(np.abs(row) > 0.0)
        rows.extend([q] * nz.size)
        cols.extend(nz.tolist())
        vals.extend(row[nz].tolist())

    return sparse.csr_matrix((vals, (rows, cols)), shape=(num_targets, num_all))


def pu_localized_evaluate(
    domain: DomainDescriptor,
    patch_data: PUPatchData,
    _xi: int,
    coeffs: np.ndarray,
    xq: np.ndarray,
) -> np.ndarray:
    xq = np.asarray(xq, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    if xq.size == 0:
        return np.zeros((0, coeffs.shape[1] if coeffs.ndim > 1 else 1))
    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]

    radius = patch_data.radius
    centers = patch_data.centers
    node_ids = patch_data.node_ids
    stencils = patch_data.stencils
    sp = patch_data.stencil_props
    patch_ids_per_query = query_patch_ids(patch_data, xq, radius)

    values = np.zeros((xq.shape[0], coeffs.shape[1]))
    weight_sum = np.zeros(xq.shape[0])
    for q in range(xq.shape[0]):
        patch_ids = patch_ids_per_query[q]
        if patch_ids.size == 0:
            continue
        center_dist = np.linalg.norm(centers[patch_ids] - xq[q], axis=1)
        alpha = pu_patch_weight(center_dist / radius)
        alpha_sum = float(alpha.sum())
        if alpha_sum <= 1.0e-14:
            alpha = np.ones_like(alpha)
            alpha_sum = float(alpha.sum())
        alpha = alpha / alpha_sum
        for k, p in enumerate(patch_ids):
            ids = node_ids[p]
            w = stencils[p].eval_weights(sp, xq[q : q + 1])[0]
            values[q] += alpha[k] * (w @ coeffs[ids])
        weight_sum[q] = 1.0

    missing = weight_sum <= 1.0e-14
    if np.any(missing):
        idx, _ = domain.query_knn("all", xq[missing], 1)
        values[missing] = coeffs[idx[:, 0]]
        weight_sum[missing] = 1.0
    return values / weight_sum[:, None]


def choose_patch_spacing(h: float, patch_spacing_factor: float) -> float:
    return float(patch_spacing_factor * h) if patch_spacing_factor > 0 else float(2.0 * h)


def choose_patch_radius(h: float, patch_radius_factor: float) -> float:
    return float(patch_radius_factor * h) if patch_radius_factor > 0 else float(3.0 * h)


def choose_patch_centers(x: np.ndarray, spacing: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros((0, x.shape[1] if x.ndim == 2 else 0))
    remaining = np.ones(x.shape[0], dtype=bool)
    centers = []
    for i in range(x.shape[0]):
        if not remaining[i]:
            continue
        xi = x[i]
        centers.append(xi.copy())
        d = np.linalg.norm(x - xi, axis=1)
        remaining[d <= spacing] = False
    return np.asarray(centers, dtype=float)


def build_patch_node_ids(
    domain: DomainDescriptor,
    centers: np.ndarray,
    radius: float,
    min_patch_nodes: int,
    tree_mode: str,
) -> list[np.ndarray]:
    patch_node_ids: list[np.ndarray] = []
    num_all = domain.get_num_total_nodes() if tree_mode == "all" else domain.get_num_int_bdry_nodes()
    for p in range(centers.shape[0]):
        ids, _ = domain.query_ball(tree_mode, centers[p], radius)
        patch_ids = np.asarray(ids[0], dtype=int)
        if patch_ids.size < min_patch_nodes:
            ids_knn, _ = domain.query_knn(tree_mode, centers[p], min(min_patch_nodes, num_all))
            patch_ids = np.asarray(ids_knn[0], dtype=int)
        patch_node_ids.append(patch_ids)
    return patch_node_ids


def choose_minimum_patch_nodes(dim: int, xi: int) -> int:
    ell = max(xi + 1, 2)
    npoly = total_degree_indices(dim, ell).shape[0]
    return 2 * npoly + 1


def build_patch_stencil_properties(dim: int, xi: int) -> StencilProperties:
    sp = StencilProperties()
    sp.dim = dim
    sp.ell = max(xi + 1, 2)
    sp.npoly = total_degree_indices(dim, sp.ell).shape[0]
    sp.spline_degree = max(5, sp.ell)
    if sp.spline_degree % 2 == 0:
        sp.spline_degree -= 1
    return sp


def build_patch_stencil(xnodes: np.ndarray, sp: StencilProperties) -> RBFStencil:
    stencil = RBFStencil()
    stencil.initialize_geometry(np.asarray(xnodes, dtype=float), sp)
    return stencil


def build_patch_stencils(xnodes: np.ndarray, node_ids: list[np.ndarray], sp: StencilProperties) -> list[RBFStencil]:
    return [build_patch_stencil(xnodes[ids], sp) for ids in node_ids]


def build_center_tree(centers: np.ndarray) -> PatchCenterTree:
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return PatchCenterTree(points=centers, searcher=None, has_searcher=False)
    return PatchCenterTree(points=centers, searcher=cKDTree(centers), has_searcher=True)


def query_patch_ids(patch_data: PUPatchData, xq: np.ndarray, radius: float) -> list[np.ndarray]:
    tree = patch_data.center_tree
    centers = patch_data.centers
    if tree.has_searcher and tree.searcher is not None:
        patch_ids_per_query = [np.asarray(ids, dtype=int) for ids in tree.searcher.query_ball_point(xq, radius)]
    else:
        d = distance_matrix(xq, centers)
        patch_ids_per_query = [np.flatnonzero(d[q] < radius) for q in range(xq.shape[0])]
    for q, patch_ids in enumerate(patch_ids_per_query):
        if patch_ids.size == 0 and centers.size != 0:
            d = np.linalg.norm(centers - xq[q], axis=1)
            patch_ids_per_query[q] = np.array([int(np.argmin(d))], dtype=int)
    return patch_ids_per_query


def local_operator_weights(
    stencil: RBFStencil,
    xq: np.ndarray,
    sp: StencilProperties,
    op_name: str,
    nr: np.ndarray,
    neu_coeff: float,
    dir_coeff: float,
) -> np.ndarray:
    xq = np.asarray(xq, dtype=float).reshape(1, -1)
    xloc = stencil.x_stencil
    xc = (xq - stencil.xm) / stencil.width
    r = distance_matrix(xq, xloc)
    op = OpProperties(nosolve=False, selectdim=0)
    name = str(op_name).lower()
    if name in {"interp", "interpolation"}:
        bpoly = stencil.basis.evaluate(xc, np.zeros((1, xloc.shape[1]), dtype=int), True).T
        b = np.vstack([stencil.phs_rbf(r, sp.spline_degree).T, bpoly])
    elif name in {"lap", "laplacian"}:
        b = stencil.lap_op(sp, op, r, xq, xloc, xc, stencil.xc)
    elif name in {"bc", "boundary"}:
        b = stencil.bc_op(sp, op, neu_coeff, dir_coeff, r, xq, xloc, xc, stencil.xc, np.asarray(nr, dtype=float).reshape(1, -1))
    else:
        raise ValueError(f'unknown local operator "{op_name}"')
    w_full = stencil.stable_solve(stencil.solve_lhs, b)
    return w_full[: xloc.shape[0], 0]


def pu_patch_weight(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    w = np.zeros_like(r)
    mask = r < 1.0
    t = 1.0 - r[mask]
    rm = r[mask]
    w[mask] = t**8 * (32.0 * rm**3 + 25.0 * rm**2 + 8.0 * rm + 1.0)
    return w


def copy_stencil_properties(sp: StencilProperties) -> StencilProperties:
    return StencilProperties(
        n=sp.n,
        dim=sp.dim,
        ell=sp.ell,
        spline_degree=sp.spline_degree,
        npoly=sp.npoly,
        width=sp.width,
        tree_mode=sp.tree_mode,
        point_set=sp.point_set,
    )
