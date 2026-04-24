from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import RBFLevelSet


def generate_poisson_nodes_in_box(
    radius_or_func: float | Callable[[np.ndarray, float], float],
    x_min: np.ndarray,
    x_max: np.ndarray,
    *,
    attempts: int = 30,
    seed: int | None = None,
    deterministic: bool | None = None,
    strip_count: int | None = None,
    use_parallel: bool = True,
    min_radius: float | None = None,
    boundary_points: np.ndarray | None = None,
    boundary_refinement_fraction: float = 1.0,
    boundary_distance: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    # Public entry point for the Matlab-style Poisson sampler. The sampler can
    # run in fixed-radius or variable-radius mode, and can optionally shrink
    # the local target radius in a boundary band.
    x_min = np.asarray(x_min, dtype=float).reshape(-1)
    x_max = np.asarray(x_max, dtype=float).reshape(-1)
    dim = x_min.size
    opts = _parse_poisson_options(
        radius_or_func,
        x_min,
        x_max,
        attempts=attempts,
        seed=seed,
        deterministic=deterministic,
        strip_count=strip_count,
        use_parallel=use_parallel,
        min_radius=min_radius,
        boundary_points=boundary_points,
        boundary_refinement_fraction=boundary_refinement_fraction,
        boundary_distance=boundary_distance,
    )
    if np.any(x_max <= x_min):
        info = _empty_poisson_info(opts, dim)
        return np.zeros((0, dim)), info

    boxes = _build_strip_boxes(x_min, x_max, float(opts["split_tol"]), int(opts["strip_count"]))
    clouds = []
    for k, box in enumerate(boxes):
        clouds.append(
            _poisson_strip_sample(
                box["sample_min"],
                box["sample_max"],
                opts,
                int((int(opts["base_seed"]) + 104729 * k) % (2**32 - 1)),
            )
        )
    points = _flatten_strip_clouds(clouds, x_min, x_max)
    info = {
        "dimension": dim,
        "mode": opts["mode"],
        "radius": opts["radius"],
        "min_radius": float(opts["min_radius"]),
        "attempts": int(opts["attempts"]),
        "seed": int(opts["seed"]),
        "deterministic": bool(opts["deterministic"]),
        "strip_count": int(opts["strip_count"]),
        "used_parallel": bool(opts["use_parallel"]),
        "boundary_refinement_fraction": float(opts["boundary_refinement_fraction"]),
        "boundary_distance": float(opts["boundary_distance"]),
        "num_points": int(points.shape[0]),
    }
    return points, info


def _parse_poisson_options(
    radius_or_func: float | Callable[[np.ndarray, float], float],
    x_min: np.ndarray,
    x_max: np.ndarray,
    *,
    attempts: int,
    seed: int | None,
    deterministic: bool | None,
    strip_count: int | None,
    use_parallel: bool,
    min_radius: float | None,
    boundary_points: np.ndarray | None,
    boundary_refinement_fraction: float,
    boundary_distance: float,
) -> dict[str, object]:
    # Normalize the mixed sampler API into one options dictionary so the strip
    # sampler can stay agnostic to whether the user requested fixed or variable
    # radii, deterministic or random seeding, and optional boundary refinement.
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if x_min.size != x_max.size:
        raise ValueError("x_min and x_max must have the same length")
    if boundary_refinement_fraction <= 0 or boundary_refinement_fraction > 1:
        raise ValueError("boundary_refinement_fraction must be in (0, 1]")
    if boundary_distance < 0:
        raise ValueError("boundary_distance must be nonnegative")

    if boundary_points is None:
        boundary_points = np.zeros((0, x_min.size))
    else:
        boundary_points = np.asarray(boundary_points, dtype=float)
        if boundary_points.ndim != 2 or boundary_points.shape[1] != x_min.size:
            raise ValueError("boundary_points must have the same column count as the box dimension")

    had_explicit_seed = seed is not None
    if seed is None:
        seed = int(np.random.randint(1, 2**31))
    if deterministic is None:
        deterministic = had_explicit_seed
    else:
        deterministic = bool(deterministic)

    requested_strip_count = None if strip_count is None else max(1, int(np.floor(strip_count)))
    actual_strip_count = _default_strip_count(bool(use_parallel), deterministic, requested_strip_count)
    used_parallel = bool(use_parallel) and actual_strip_count > 1
    has_boundary_refinement = (
        boundary_points.shape[0] > 0
        and boundary_refinement_fraction < 1.0
        and boundary_distance > 0.0
    )

    if callable(radius_or_func):
        if min_radius is None:
            raise ValueError("variable-density sampling requires min_radius")
        min_radius = float(min_radius)
        if min_radius <= 0:
            raise ValueError("min_radius must be positive")
        radius = float("nan")
        if has_boundary_refinement:
            mode = "variable_radius_with_boundary_refinement"
            grid_radius = boundary_refinement_fraction * min_radius
        else:
            mode = "variable_radius"
            grid_radius = min_radius
        rad_func = radius_or_func
    else:
        radius = float(radius_or_func)
        if radius <= 0:
            raise ValueError("radius must be positive")
        min_radius = radius if min_radius is None else float(min_radius)
        if has_boundary_refinement:
            mode = "fixed_radius_with_boundary_refinement"
            grid_radius = boundary_refinement_fraction * radius
        else:
            mode = "fixed_radius"
            grid_radius = radius
        rad_func = None

    return {
        "radius": radius,
        "rad_func": rad_func,
        "min_radius": float(min_radius),
        "attempts": int(attempts),
        "seed": int(seed),
        "base_seed": int(seed) % (2**32),
        "deterministic": deterministic,
        "strip_count": actual_strip_count,
        "use_parallel": used_parallel,
        "mode": mode,
        "boundary_points": boundary_points,
        "boundary_refinement_fraction": float(boundary_refinement_fraction),
        "boundary_distance": float(boundary_distance),
        "has_boundary_refinement": has_boundary_refinement,
        "boundary_tree": cKDTree(boundary_points) if boundary_points.shape[0] > 0 else None,
        "grid_radius": float(grid_radius),
        "split_tol": float(min_radius),
    }


def _empty_poisson_info(opts: dict[str, object], dim: int) -> dict[str, object]:
    return {
        "dimension": dim,
        "mode": opts["mode"],
        "radius": opts["radius"],
        "min_radius": float(opts["min_radius"]),
        "attempts": int(opts["attempts"]),
        "seed": int(opts["seed"]),
        "deterministic": bool(opts["deterministic"]),
        "strip_count": int(opts["strip_count"]),
        "used_parallel": False,
        "boundary_refinement_fraction": float(opts["boundary_refinement_fraction"]),
        "boundary_distance": float(opts["boundary_distance"]),
        "num_points": 0,
    }


def _default_strip_count(use_parallel: bool, deterministic: bool, requested_strip_count: int | None) -> int:
    if deterministic:
        return 1
    if requested_strip_count is not None:
        return requested_strip_count
    if not use_parallel:
        return 1
    return 1


def _build_strip_boxes(x_min: np.ndarray, x_max: np.ndarray, split_tol: float, strip_count: int) -> list[dict[str, np.ndarray]]:
    # Split the box along the first coordinate using the same slightly shrunken
    # strip extent as Matlab. Deterministic runs collapse to one strip.
    dim0 = (x_max[0] - x_min[0]) / strip_count
    out = []
    for k in range(strip_count):
        sample_min = x_min.copy()
        sample_max = x_max.copy()
        sample_min[0] = x_min[0] + k * dim0
        sample_max[0] = min(x_max[0], sample_min[0] + dim0 - 0.33 * split_tol)
        out.append({"sample_min": sample_min, "sample_max": sample_max})
    return out


def _poisson_strip_sample(
    sample_min: np.ndarray,
    sample_max: np.ndarray,
    opts: dict[str, object],
    seed: int,
) -> np.ndarray:
    # Bridson-style dart throwing on one strip, with a local radius determined
    # on the fly from the active point and the configured refinement mode.
    if np.any(sample_max <= sample_min):
        return np.zeros((0, sample_min.size))

    dim = sample_min.size
    cell_size = float(opts["grid_radius"]) / np.sqrt(dim)
    grid_size = np.maximum(1, np.ceil((sample_max - sample_min) / cell_size).astype(int))
    grid = np.full(tuple(int(v) for v in grid_size), -1, dtype=int)
    points: list[np.ndarray] = []
    active: list[int] = []
    rng = np.random.Generator(np.random.MT19937(int(seed)))

    x0 = sample_min + rng.random(dim) * (sample_max - sample_min)
    points.append(x0)
    active.append(0)
    grid[_point_to_cell(x0, sample_min, cell_size)] = 0

    while active:
        pick = int(rng.integers(len(active)))
        active_idx = active[pick]
        base = points[active_idx]
        active_radius = _local_radius(base, opts)
        accepted = False
        for _ in range(int(opts["attempts"])):
            candidate = _propose_candidate(base, active_radius, rng)
            if not _point_in_box(candidate, sample_min, sample_max):
                continue
            if _has_conflicting_neighbor(candidate, active_idx, points, grid, sample_min, cell_size, grid_size, opts):
                continue
            idx = len(points)
            points.append(candidate)
            active.append(idx)
            grid[_point_to_cell(candidate, sample_min, cell_size)] = idx
            accepted = True
            break
        if not accepted:
            active[pick] = active[-1]
            active.pop()

    return np.asarray(points, dtype=float)


def _propose_candidate(base: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    dim = base.size
    direction = rng.normal(size=dim)
    norm_sq = float(np.dot(direction, direction))
    inv_norm = 1.0 / max(np.sqrt(norm_sq), np.finfo(float).eps)
    direction = direction * inv_norm
    shell_radius = radius * (1.0 + rng.random() * (2**dim - 1)) ** (1.0 / dim)
    return base + shell_radius * direction


def _point_in_box(point: np.ndarray, sample_min: np.ndarray, sample_max: np.ndarray) -> bool:
    for d in range(point.size):
        if point[d] < sample_min[d] or point[d] > sample_max[d]:
            return False
    return True


def _local_radius(point: np.ndarray, opts: dict[str, object]) -> float:
    mode = str(opts["mode"])
    if mode == "fixed_radius":
        return float(opts["radius"])
    if mode == "fixed_radius_with_boundary_refinement":
        return _boundary_refined_radius(point, opts)
    if mode == "variable_radius":
        radius = float(opts["rad_func"](point, float(opts["min_radius"])))
        return radius if radius > 1e-10 else float(opts["min_radius"])
    if mode == "variable_radius_with_boundary_refinement":
        base_radius = float(opts["rad_func"](point, float(opts["min_radius"])))
        if base_radius <= 1e-10:
            base_radius = float(opts["min_radius"])
        if _boundary_rad_frac(point, opts) < 1.0:
            return float(opts["boundary_refinement_fraction"]) * float(opts["min_radius"])
        return base_radius
    raise ValueError("unknown Poisson sampling mode")


def _boundary_rad_frac(point: np.ndarray, opts: dict[str, object]) -> float:
    if not bool(opts["has_boundary_refinement"]):
        return 1.0
    dist = _nearest_boundary_distance(
        point,
        np.asarray(opts["boundary_points"], dtype=float),
        opts.get("boundary_tree"),
    )
    if dist <= float(opts["boundary_distance"]):
        return float(opts["boundary_refinement_fraction"])
    return 1.0


def _boundary_refined_radius(point: np.ndarray, opts: dict[str, object]) -> float:
    return _boundary_rad_frac(point, opts) * float(opts["radius"])


def _has_conflicting_neighbor(
    point: np.ndarray,
    active_idx: int,
    points: list[np.ndarray],
    grid: np.ndarray,
    x_min: np.ndarray,
    cell_size: float,
    grid_size: np.ndarray,
    opts: dict[str, object],
) -> bool:
    # Neighbor rejection rules differ slightly between the plain and
    # boundary-refined modes. The refined fixed-radius mode checks a symmetric
    # pairwise threshold to preserve the near-boundary radius reduction.
    candidate_radius = _local_radius(point, opts)
    mode = str(opts["mode"])
    if mode == "fixed_radius_with_boundary_refinement":
        radius_for_reach = max(candidate_radius, float(opts["radius"]))
        exclude_active = False
        pairwise = True
    elif mode in {"fixed_radius", "variable_radius", "variable_radius_with_boundary_refinement"}:
        radius_for_reach = candidate_radius
        exclude_active = True
        pairwise = False
    else:
        raise ValueError("unknown Poisson sampling mode")

    idx = _point_to_cell(point, x_min, cell_size)
    reach = max(1, int(np.ceil(radius_for_reach / cell_size)))
    if point.size == 2:
        x0_lo = max(0, idx[0] - reach)
        x0_hi = min(int(grid_size[0]) - 1, idx[0] + reach)
        x1_lo = max(0, idx[1] - reach)
        x1_hi = min(int(grid_size[1]) - 1, idx[1] + reach)
        for i0 in range(x0_lo, x0_hi + 1):
            for i1 in range(x1_lo, x1_hi + 1):
                j = int(grid[i0, i1])
                if j < 0:
                    continue
                if exclude_active and j == active_idx:
                    continue
                if pairwise:
                    threshold = max(candidate_radius, _local_radius(points[j], opts))
                else:
                    threshold = candidate_radius
                if _squared_distance(point, points[j]) < threshold * threshold:
                    return True
        return False
    if point.size == 3:
        x0_lo = max(0, idx[0] - reach)
        x0_hi = min(int(grid_size[0]) - 1, idx[0] + reach)
        x1_lo = max(0, idx[1] - reach)
        x1_hi = min(int(grid_size[1]) - 1, idx[1] + reach)
        x2_lo = max(0, idx[2] - reach)
        x2_hi = min(int(grid_size[2]) - 1, idx[2] + reach)
        for i0 in range(x0_lo, x0_hi + 1):
            for i1 in range(x1_lo, x1_hi + 1):
                for i2 in range(x2_lo, x2_hi + 1):
                    j = int(grid[i0, i1, i2])
                    if j < 0:
                        continue
                    if exclude_active and j == active_idx:
                        continue
                    if pairwise:
                        threshold = max(candidate_radius, _local_radius(points[j], opts))
                    else:
                        threshold = candidate_radius
                    if _squared_distance(point, points[j]) < threshold * threshold:
                        return True
        return False
    ranges = [range(max(0, idx[d] - reach), min(int(grid_size[d]) - 1, idx[d] + reach) + 1) for d in range(point.size)]
    for cell in product(*ranges):
        zero_cell = tuple(int(v) for v in cell)
        j = int(grid[zero_cell])
        if j < 0:
            continue
        if exclude_active and j == active_idx:
            continue
        if pairwise:
            threshold = max(candidate_radius, _local_radius(points[j], opts))
        else:
            threshold = candidate_radius
        if _squared_distance(point, points[j]) < threshold * threshold:
            return True
    return False


def _nearest_boundary_distance(point: np.ndarray, boundary_points: np.ndarray, boundary_tree: cKDTree | None = None) -> float:
    if boundary_points.size == 0:
        return float("inf")
    tree = boundary_tree if boundary_tree is not None else cKDTree(boundary_points)
    dist, _ = tree.query(point, k=1)
    return float(dist)


def _flatten_strip_clouds(local_clouds: list[np.ndarray], x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    if not local_clouds:
        return np.zeros((0, x_min.size))
    any_points = [pts for pts in local_clouds if pts.size]
    if not any_points:
        return np.zeros((0, x_min.size))
    points = np.vstack(any_points)
    in_box = np.all((points >= x_min) & (points <= x_max), axis=1)
    return points[in_box]


def _point_to_cell(point: np.ndarray, x_min: np.ndarray, cell_size: float) -> tuple[int, ...]:
    return tuple(max(0, int(np.floor((point[d] - x_min[d]) / cell_size))) for d in range(point.size))


def _squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return float(np.dot(diff, diff))


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
    # Filter a raw node cloud against the level set of a geometry object. This
    # is the point where the box Poisson cloud becomes a geometry-conforming
    # interior or exterior point set.
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
        radius_function: Callable[[np.ndarray, float], float] | None = None,
        do_outer_refinement: bool = False,
        outer_fraction_of_h: float = 1.0,
        outer_refinement_zone_size_as_multiple_of_h: float = 2.0,
        **kwargs: object,
    ) -> None:
        # Build boundary data first so the box sampler can optionally use those
        # points to drive near-boundary refinement, then clip the raw cloud
        # against the level set.
        xb, nrmls, level_set = _build_boundary_state(geometry)
        x_min, x_max = bounding_box_extents(geometry, True)
        sampler_input = radius if radius_function is None else radius_function
        sampler_kwargs = dict(kwargs)
        sampler_kwargs["min_radius"] = radius
        if do_outer_refinement:
            sampler_kwargs["boundary_points"] = xb
            sampler_kwargs["boundary_refinement_fraction"] = outer_fraction_of_h
            sampler_kwargs["boundary_distance"] = outer_refinement_zone_size_as_multiple_of_h * radius

        x_raw, info = generate_poisson_nodes_in_box(sampler_input, x_min, x_max, **sampler_kwargs)
        self.xi = x_raw
        self.xb = np.zeros((0, x_raw.shape[1]))
        self.xg = np.zeros((0, x_raw.shape[1]))
        self.nrmls = np.zeros((0, x_raw.shape[1]))
        self.xi_orig = x_raw
        self.xi_pds_raw = x_raw
        self.s_dim = x_raw.shape[1]
        self.last_info = info

        clearance = outer_fraction_of_h * radius
        self.clip_to_geometry(geometry, keep="inside", boundary_clearance=clearance)
        self.last_info["min_active_radius"] = clearance
        self.last_info["outer_refinement"] = {
            "enabled": bool(do_outer_refinement),
            "refinement_fraction": outer_fraction_of_h,
            "zone_size": outer_refinement_zone_size_as_multiple_of_h * radius,
            "boundary_distance": outer_refinement_zone_size_as_multiple_of_h * radius,
        }
        self.last_info["boundary_node_count"] = xb.shape[0]
        self.last_info["boundary_level_set_built"] = level_set is not None

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
        # Promote the generated interior cloud into the full descriptor used by
        # the solver/assembly layers by adding boundary and ghost nodes plus the
        # relevant level-set metadata.
        self.generate_interior_nodes_from_geometry(geometry, radius, **kwargs)
        xb, nrmls, level_set = _build_boundary_state(geometry)
        xg = xb + 0.5 * radius * nrmls
        descriptor = DomainDescriptor()
        descriptor.set_nodes(self.xi, xb, xg)
        descriptor.set_normals(nrmls)
        descriptor.set_sep_rad(float(radius))
        descriptor.set_outer_level_set(level_set)
        descriptor.set_boundary_level_sets([level_set])
        descriptor.build_structs()
        self.xb = xb
        self.xg = xg
        self.nrmls = nrmls
        self.descriptor = descriptor
        self.last_info["boundary_node_count"] = xb.shape[0]
        self.last_info["ghost_node_count"] = xg.shape[0]
        self.last_info["sep_radius"] = radius
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
