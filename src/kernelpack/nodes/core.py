from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import RBFLevelSet, distance_matrix


def generate_poisson_nodes_in_box(
    radius: float,
    x_min: np.ndarray,
    x_max: np.ndarray,
    *,
    attempts: int = 30,
    seed: int | None = None,
    deterministic: bool | None = None,
    strip_count: int | None = None,
    use_parallel: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    x_min = np.asarray(x_min, dtype=float).reshape(-1)
    x_max = np.asarray(x_max, dtype=float).reshape(-1)
    dim = x_min.size
    if np.any(x_max <= x_min):
        info = {
            "dimension": dim,
            "radius": radius,
            "attempts": attempts,
            "seed": seed,
            "deterministic": bool(seed is not None if deterministic is None else deterministic),
            "strip_count": 1 if strip_count is None else strip_count,
            "used_parallel": False,
            "num_candidates": 0,
            "num_points": 0,
        }
        return np.zeros((0, dim)), info
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    if deterministic is None:
        deterministic = True
    if strip_count is None:
        strip_count = 5 if deterministic else 1

    boxes = _build_strip_boxes(x_min, x_max, radius, int(strip_count))
    clouds = []
    for k, box in enumerate(boxes):
        clouds.append(
            _bridson_poisson_box(
                radius,
                box["sample_min"],
                box["sample_max"],
                attempts,
                (seed + 104729 * k) % (2**32 - 1),
            )
        )
    merged = _merge_local_clouds(clouds, x_min, x_max, radius)
    info = {
        "dimension": dim,
        "radius": radius,
        "attempts": attempts,
        "seed": seed,
        "deterministic": bool(deterministic),
        "strip_count": int(strip_count),
        "used_parallel": False,
        "num_candidates": int(merged["num_candidates"]),
        "num_points": int(merged["points"].shape[0]),
    }
    return merged["points"], info


def _build_strip_boxes(x_min: np.ndarray, x_max: np.ndarray, radius: float, strip_count: int) -> list[dict[str, np.ndarray]]:
    dim0 = (x_max[0] - x_min[0]) / strip_count
    overlap = radius
    out = []
    for k in range(strip_count):
        core_min = x_min.copy()
        core_max = x_max.copy()
        core_min[0] = x_min[0] + k * dim0
        core_max[0] = x_min[0] + (k + 1) * dim0
        sample_min = core_min.copy()
        sample_max = core_max.copy()
        if k > 0:
            sample_min[0] = max(x_min[0], sample_min[0] - overlap)
        if k < strip_count - 1:
            sample_max[0] = min(x_max[0], sample_max[0] + overlap)
        out.append({"sample_min": sample_min, "sample_max": sample_max})
    return out


def _bridson_poisson_box(radius: float, x_min: np.ndarray, x_max: np.ndarray, attempts: int, seed: int) -> np.ndarray:
    dim = x_min.size
    rng = np.random.default_rng(seed)
    cell_size = radius / np.sqrt(dim)
    grid_size = np.maximum(1, np.ceil((x_max - x_min) / cell_size).astype(int))
    grid: dict[tuple[int, ...], int] = {}
    points: list[np.ndarray] = []
    active: list[int] = []

    x0 = x_min + rng.random(dim) * (x_max - x_min)
    points.append(x0)
    active.append(0)
    grid[_point_to_cell(x0, x_min, cell_size)] = 0

    while active:
        pick = int(rng.integers(len(active)))
        active_idx = active[pick]
        base = points[active_idx]
        accepted = False
        for _ in range(attempts):
            direction = rng.normal(size=dim)
            norm = np.linalg.norm(direction)
            direction = direction / (norm if norm > 0 else 1.0)
            shell_radius = radius * (1.0 + rng.random() * (2**dim - 1)) ** (1.0 / dim)
            candidate = base + shell_radius * direction
            if np.any(candidate < x_min) or np.any(candidate > x_max):
                continue
            if _has_neighbor(candidate, radius, points, grid, x_min, cell_size, grid_size):
                continue
            idx = len(points)
            points.append(candidate)
            active.append(idx)
            grid[_point_to_cell(candidate, x_min, cell_size)] = idx
            accepted = True
            break
        if not accepted:
            active[pick] = active[-1]
            active.pop()
    return np.asarray(points, dtype=float)


def _point_to_cell(point: np.ndarray, x_min: np.ndarray, cell_size: float) -> tuple[int, ...]:
    return tuple(np.maximum(1, np.floor((point - x_min) / cell_size).astype(int) + 1))


def _has_neighbor(
    point: np.ndarray,
    radius: float,
    points: list[np.ndarray],
    grid: dict[tuple[int, ...], int],
    x_min: np.ndarray,
    cell_size: float,
    grid_size: np.ndarray,
) -> bool:
    idx = np.asarray(_point_to_cell(point, x_min, cell_size))
    reach = max(1, int(np.ceil(radius / cell_size)))
    ranges = [range(max(1, idx[d] - reach), min(grid_size[d], idx[d] + reach) + 1) for d in range(idx.size)]
    for cell in np.array(np.meshgrid(*ranges)).T.reshape(-1, idx.size):
        key = tuple(int(v) for v in cell)
        if key not in grid:
            continue
        j = grid[key]
        if np.linalg.norm(point - points[j]) < radius:
            return True
    return False


def _merge_local_clouds(local_clouds: list[np.ndarray], x_min: np.ndarray, x_max: np.ndarray, radius: float) -> dict[str, object]:
    if not local_clouds:
        return {"points": np.zeros((0, x_min.size)), "num_candidates": 0}
    all_points = np.vstack([pts for pts in local_clouds if pts.size]) if any(pts.size for pts in local_clouds) else np.zeros((0, x_min.size))
    if all_points.size == 0:
        return {"points": all_points, "num_candidates": 0}
    in_box = np.all((all_points >= x_min) & (all_points <= x_max), axis=1)
    all_points = np.array(sorted(map(tuple, all_points[in_box])), dtype=float)
    return {"points": _accept_by_global_grid(all_points, x_min, x_max, radius), "num_candidates": all_points.shape[0]}


def _accept_by_global_grid(points: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, radius: float) -> np.ndarray:
    dim = points.shape[1]
    if points.shape[0] == 0:
        return points
    cell_size = radius / np.sqrt(dim)
    grid_size = np.maximum(1, np.ceil((x_max - x_min) / cell_size).astype(int))
    grid: dict[tuple[int, ...], int] = {}
    accepted: list[np.ndarray] = []
    for pt in points:
        if _has_neighbor(pt, radius, accepted, grid, x_min, cell_size, grid_size):
            continue
        grid[_point_to_cell(pt, x_min, cell_size)] = len(accepted)
        accepted.append(pt)
    return np.asarray(accepted, dtype=float)


def clip_points_by_geometry(
    x: np.ndarray,
    geometry: object,
    *,
    keep: str = "inside",
    tolerance: float = 0.0,
    boundary_clearance: float = 0.0,
    min_signed_distance: float = -np.inf,
    max_signed_distance: float = np.inf,
    auto_build_level_set: bool = True,
    use_parallel: bool = True,
    chunk_size: int = 5000,
    min_parallel_points: int = 20000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del use_parallel, chunk_size, min_parallel_points
    x = np.asarray(x, dtype=float)
    level_set = _ensure_geometry_level_set(geometry, auto_build_level_set)
    phi = level_set.evaluate(x)
    if keep.lower() == "inside":
        keep_mask = phi <= (tolerance - boundary_clearance)
    elif keep.lower() == "outside":
        keep_mask = phi >= (boundary_clearance - tolerance)
    else:
        raise ValueError("keep must be inside or outside")
    keep_mask &= phi >= min_signed_distance
    keep_mask &= phi <= max_signed_distance
    return x[keep_mask], keep_mask, phi


def _ensure_geometry_level_set(geometry: object, auto_build: bool) -> RBFLevelSet:
    if not hasattr(geometry, "get_level_set"):
        raise ValueError("geometry object must provide get_level_set")
    level_set = geometry.get_level_set()
    needs_build = not isinstance(level_set, RBFLevelSet) or level_set.n == 0
    if needs_build:
        if not auto_build:
            raise ValueError("geometry level set is not built")
        if hasattr(geometry, "build_level_set_from_geometric_model"):
            geometry.build_level_set_from_geometric_model(None)
        elif hasattr(geometry, "build_level_set"):
            geometry.build_level_set()
        else:
            raise ValueError("geometry cannot build a level set")
        level_set = geometry.get_level_set()
    return level_set


def bounding_box_extents(geometry: object, prefer_uniform: bool = True) -> tuple[np.ndarray, np.ndarray]:
    box = None
    if prefer_uniform and hasattr(geometry, "get_uniform_bounding_box"):
        box = geometry.get_uniform_bounding_box()
        if box is None or np.asarray(box).size == 0:
            if hasattr(geometry, "compute_uniform_bounding_box"):
                geometry.compute_uniform_bounding_box()
                box = geometry.get_uniform_bounding_box()
    if (box is None or np.asarray(box).size == 0) and hasattr(geometry, "get_bounding_box"):
        box = geometry.get_bounding_box()
        if box is None or np.asarray(box).size == 0:
            if hasattr(geometry, "compute_bounding_box"):
                geometry.compute_bounding_box()
                box = geometry.get_bounding_box()
    if box is None or np.asarray(box).size == 0:
        if prefer_uniform and hasattr(geometry, "get_uniform_sample_sites"):
            box = geometry.get_uniform_sample_sites()
        elif hasattr(geometry, "get_sample_sites"):
            box = geometry.get_sample_sites()
        elif hasattr(geometry, "get_bdry_nodes"):
            box = geometry.get_bdry_nodes()
        else:
            raise ValueError("geometry does not expose bounding data")
    box = np.asarray(box, dtype=float)
    return box.min(axis=0), box.max(axis=0)


@dataclass
class DomainNodeGenerator:
    xi: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xg: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    nrmls: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xi_orig: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xi_pds_raw: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    s_dim: int = 0
    last_info: dict[str, object] = field(default_factory=dict)
    descriptor: DomainDescriptor = field(default_factory=DomainDescriptor)

    def generate_poisson_nodes(self, radius: float, x_min: np.ndarray, x_max: np.ndarray, **kwargs: object) -> None:
        x, info = generate_poisson_nodes_in_box(radius, x_min, x_max, **kwargs)
        self.xi = x
        self.xb = np.zeros((0, x.shape[1]))
        self.xg = np.zeros((0, x.shape[1]))
        self.nrmls = np.zeros((0, x.shape[1]))
        self.xi_orig = x
        self.xi_pds_raw = x
        self.s_dim = x.shape[1]
        self.last_info = info
        self.descriptor = DomainDescriptor()

    def generate_interior_nodes_from_geometry(
        self,
        geometry: object,
        radius: float,
        *,
        do_outer_refinement: bool = False,
        outer_fraction_of_h: float = 0.5,
        outer_refinement_zone_size_as_multiple_of_h: float = 2.0,
        **kwargs: object,
    ) -> None:
        x_min, x_max = bounding_box_extents(geometry, True)
        self.generate_poisson_nodes(radius, x_min, x_max, **kwargs)
        if not do_outer_refinement:
            self.clip_to_geometry(geometry, keep="inside", boundary_clearance=radius)
            self.last_info["min_active_radius"] = radius
            return
        fine_radius = outer_fraction_of_h * radius
        zone_size = outer_refinement_zone_size_as_multiple_of_h * radius
        min_radius_used = min(radius, fine_radius)
        level_set = geometry.get_level_set()
        if not isinstance(level_set, RBFLevelSet) or level_set.n == 0:
            if hasattr(geometry, "build_level_set_from_geometric_model"):
                geometry.build_level_set_from_geometric_model(None)
            else:
                geometry.build_level_set()
            level_set = geometry.get_level_set()
        coarse_phi = level_set.evaluate(self.xi_pds_raw)
        coarse_mask = (coarse_phi <= -zone_size) & (coarse_phi <= -min_radius_used)
        coarse_nodes = self.xi_pds_raw[coarse_mask]
        fine_raw, fine_info = generate_poisson_nodes_in_box(fine_radius, x_min, x_max, **kwargs)
        fine_nodes, fine_mask, fine_phi = clip_points_by_geometry(
            fine_raw,
            geometry,
            keep="inside",
            boundary_clearance=min_radius_used,
            min_signed_distance=-zone_size,
            max_signed_distance=-min_radius_used,
        )
        self.xi_pds_raw = np.vstack([self.xi_pds_raw, fine_raw])
        self.xi = np.vstack([coarse_nodes, fine_nodes])
        self.xi_orig = self.xi
        self.s_dim = self.xi.shape[1]
        self.last_info["outer_refinement"] = {
            "enabled": True,
            "coarse_radius": radius,
            "fine_radius": fine_radius,
            "zone_size": zone_size,
            "min_active_radius": min_radius_used,
            "coarse_points_kept": coarse_nodes.shape[0],
            "fine_raw_points": fine_raw.shape[0],
            "fine_points_kept": fine_nodes.shape[0],
            "fine_info": fine_info,
            "fine_mask": fine_mask,
            "fine_phi": fine_phi[fine_mask],
        }
        self.last_info["min_active_radius"] = min_radius_used

    def clip_to_geometry(self, geometry: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, mask, phi = clip_points_by_geometry(self.xi_pds_raw, geometry, **kwargs)
        self.xi = x
        self.xi_orig = x
        self.s_dim = x.shape[1]
        self.last_info["clip_mask"] = mask
        self.last_info["clip_count"] = x.shape[0]
        self.last_info["clip_levelset_values"] = phi[mask]
        return x, mask, phi

    def build_domain_descriptor_from_geometry(self, geometry: object, radius: float, **kwargs: object) -> DomainDescriptor:
        self.generate_interior_nodes_from_geometry(geometry, radius, **kwargs)
        xb, nrmls, level_set = _build_boundary_state(geometry)
        min_radius_used = self.last_info.get("min_active_radius", radius)
        xg = xb + 0.5 * min_radius_used * nrmls
        descriptor = DomainDescriptor()
        descriptor.set_nodes(self.xi, xb, xg)
        descriptor.set_normals(nrmls)
        descriptor.set_sep_rad(float(min_radius_used))
        descriptor.set_outer_level_set(level_set)
        descriptor.set_boundary_level_sets([level_set])
        descriptor.build_structs()
        self.xb = xb
        self.xg = xg
        self.nrmls = nrmls
        self.descriptor = descriptor
        self.last_info["boundary_node_count"] = xb.shape[0]
        self.last_info["ghost_node_count"] = xg.shape[0]
        self.last_info["min_active_radius"] = min_radius_used
        return descriptor

    def get_interior_nodes(self) -> np.ndarray:
        return self.xi

    def get_bdry_nodes(self) -> np.ndarray:
        return self.xb

    def get_ghost_nodes(self) -> np.ndarray:
        return self.xg

    def get_nrmls(self) -> np.ndarray:
        return self.nrmls

    def get_raw_poisson_interior_nodes(self) -> np.ndarray:
        return self.xi_pds_raw

    def get_domain_descriptor(self) -> DomainDescriptor:
        return self.descriptor


def _build_boundary_state(geometry: object) -> tuple[np.ndarray, np.ndarray, RBFLevelSet]:
    if hasattr(geometry, "get_uniform_bdry_nodes"):
        xb = geometry.get_uniform_bdry_nodes()
        nrmls = geometry.get_uniform_bdry_nrmls()
    elif hasattr(geometry, "get_uniform_sample_sites"):
        xb = geometry.get_uniform_sample_sites()
        nrmls = geometry.get_uniform_nrmls()
    elif hasattr(geometry, "get_bdry_nodes"):
        xb = geometry.get_bdry_nodes()
        nrmls = geometry.get_bdry_nrmls()
    else:
        xb = geometry.get_sample_sites()
        nrmls = geometry.get_nrmls()
    level_set = geometry.get_level_set()
    if not isinstance(level_set, RBFLevelSet) or level_set.n == 0:
        if hasattr(geometry, "build_level_set_from_geometric_model"):
            geometry.build_level_set_from_geometric_model(None)
        else:
            geometry.build_level_set()
        level_set = geometry.get_level_set()
    return np.asarray(xb, dtype=float), np.asarray(nrmls, dtype=float), level_set
