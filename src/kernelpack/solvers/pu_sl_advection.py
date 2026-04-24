from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from kernelpack.domain import DomainDescriptor
from kernelpack.poly import total_degree_indices
from kernelpack.rbffd import RBFStencil, StencilProperties

from ._pu import (
    PatchCenterTree,
    build_center_tree,
    build_patch_node_ids,
    build_patch_stencil,
    choose_minimum_patch_nodes,
    choose_patch_centers,
    choose_patch_radius,
    choose_patch_spacing,
    pu_patch_weight,
    query_patch_ids,
)


def _build_patch_stencil_properties(dim: int, xi: int) -> StencilProperties:
    sp = StencilProperties()
    sp.dim = dim
    sp.ell = max(xi + 1, 2)
    sp.npoly = total_degree_indices(dim, sp.ell).shape[0]
    sp.spline_degree = max(5, 2 * ((sp.ell + 1) // 2) - 1)
    return sp


@dataclass
class PUSLAdvectionSolver:
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    dt: float = 0.0
    output_nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    output_range: tuple[int, int] = (1, 0)
    patch_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    patch_radius: float = 0.0
    patch_spacing: float = 0.0
    min_patch_nodes: int = 0
    patch_node_ids: list[np.ndarray] = field(default_factory=list)
    patch_center_tree: PatchCenterTree = field(default_factory=PatchCenterTree)
    patch_stencil_props: StencilProperties = field(default_factory=StencilProperties)
    patch_stencils: list[RBFStencil] = field(default_factory=list)
    domain_measure: float = 0.0
    boundary_condition: dict[str, object] = field(
        default_factory=lambda: {
            "mode": "unspecified",
            "normal_velocity_tolerance": 1.0e-10,
            "periodic_patches": [],
            "inflow_value": None,
        }
    )
    inflow_forward_fallback_reported: bool = False
    solve_stats: dict[str, int] = field(
        default_factory=lambda: {
            "moved_solves_zero_defect": 0,
            "moved_solves_one_defect": 0,
            "moved_solves_two_defect": 0,
            "defect_correction_solves": 0,
        }
    )

    def clear_advection_boundary_condition(self) -> None:
        self.boundary_condition = {
            "mode": "unspecified",
            "normal_velocity_tolerance": 1.0e-10,
            "periodic_patches": [],
            "inflow_value": None,
        }

    def set_tangential_flow_boundary(self, normal_velocity_tolerance: float = 1.0e-10) -> None:
        self.boundary_condition = {
            "mode": "tangential",
            "normal_velocity_tolerance": normal_velocity_tolerance,
            "periodic_patches": [],
            "inflow_value": None,
        }

    def set_periodic_boundary(self, periodic_patches: object, normal_velocity_tolerance: float = 1.0e-10) -> None:
        self.boundary_condition = {
            "mode": "periodic",
            "normal_velocity_tolerance": normal_velocity_tolerance,
            "periodic_patches": periodic_patches,
            "inflow_value": None,
        }

    def set_inflow_dirichlet_boundary(self, inflow_value: Callable[..., np.ndarray], normal_velocity_tolerance: float = 1.0e-10) -> None:
        self.boundary_condition = {
            "mode": "inflow_dirichlet",
            "normal_velocity_tolerance": normal_velocity_tolerance,
            "periodic_patches": [],
            "inflow_value": inflow_value,
        }

    def get_advection_boundary_condition(self) -> dict[str, object]:
        return self.boundary_condition

    def returns_distributed_state(self) -> bool:
        return False

    def get_output_range(self) -> tuple[int, int]:
        return self.output_range

    def init(
        self,
        domain: DomainDescriptor,
        xi: int,
        dlt: float,
        patch_spacing_factor: float = 0.0,
        patch_radius_factor: float = 0.0,
    ) -> None:
        self.domain = domain
        self.domain.build_structs()
        self.xi = xi
        self.dt = dlt
        self.output_nodes = self.domain.get_int_bdry_nodes()
        self.output_range = (1, self.output_nodes.shape[0])

        h = self.domain.get_sep_rad()
        self.patch_spacing = choose_patch_spacing(h, patch_spacing_factor)
        self.patch_radius = choose_patch_radius(h, patch_radius_factor)
        self.min_patch_nodes = choose_minimum_patch_nodes(self.output_nodes.shape[1], xi)
        self.patch_centers = choose_patch_centers(self.output_nodes, self.patch_spacing)
        self.patch_node_ids = build_patch_node_ids(self.domain, self.patch_centers, self.patch_radius, self.min_patch_nodes, "interior_boundary")
        self.patch_center_tree = build_center_tree(self.patch_centers)
        self.patch_stencil_props = _build_patch_stencil_properties(self.output_nodes.shape[1], self.xi)
        self.patch_stencils = [build_patch_stencil(self.output_nodes[ids], self.patch_stencil_props) for ids in self.patch_node_ids]
        self.domain_measure = estimate_domain_measure(self.domain)
        self.inflow_forward_fallback_reported = False

    def get_output_nodes(self) -> np.ndarray:
        return self.output_nodes

    def project_initial(self, rho0: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        coeffs = np.zeros((self.output_nodes.shape[0], 1))
        for i in range(self.output_nodes.shape[0]):
            coeffs[i, 0] = np.asarray(rho0(self.output_nodes[i]), dtype=float).reshape(-1)[0]
        return coeffs

    def project_constant(self, value: float, n_cols: int = 1) -> np.ndarray:
        return value * np.ones((self.output_nodes.shape[0], n_cols))

    def project_samples(self, nodal_samples: np.ndarray) -> np.ndarray:
        return np.asarray(nodal_samples, dtype=float)

    def evaluate_at_nodes(self, coeffs: np.ndarray) -> np.ndarray:
        return np.asarray(coeffs, dtype=float)

    def evaluate_at_points(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        return localized_evaluate(self, coeffs, x)

    def backward_sl_step(
        self,
        tn: float,
        coeffs_old: np.ndarray,
        velocity: Callable[[float, np.ndarray], np.ndarray],
        rk: Callable[[float, np.ndarray, float, Callable[[float, np.ndarray], np.ndarray]], np.ndarray] | None = None,
    ) -> np.ndarray:
        validate_boundary_condition_configuration(self)
        bc = self.boundary_condition
        t_arrival = tn + self.dt
        if bc["mode"] == "tangential":
            validate_tangential_boundary_flow(self, tn, velocity)
            validate_tangential_boundary_flow(self, t_arrival, velocity)

        xdep = trace_points_backward(self.output_nodes, tn, self.dt, velocity, rk)
        return sample_backward_traced_values(self, coeffs_old, self.output_nodes, xdep, t_arrival, velocity)

    def forward_sl_step(
        self,
        tn: float,
        coeffs_old: np.ndarray,
        velocity: Callable[[float, np.ndarray], np.ndarray],
        rk: Callable[[float, np.ndarray, float, Callable[[float, np.ndarray], np.ndarray]], np.ndarray] | None = None,
    ) -> np.ndarray:
        validate_boundary_condition_configuration(self)
        bc = self.boundary_condition
        if bc["mode"] == "inflow_dirichlet":
            if not self.inflow_forward_fallback_reported:
                self.inflow_forward_fallback_reported = True
            return self.backward_sl_step(tn, coeffs_old, velocity, rk)

        coeffs_old = np.asarray(coeffs_old, dtype=float)
        nodal_old = self.evaluate_at_nodes(coeffs_old)
        if nodal_old.ndim == 1:
            nodal_old = nodal_old[:, None]

        if bc["mode"] == "tangential":
            validate_tangential_boundary_flow(self, tn, velocity)
            validate_tangential_boundary_flow(self, tn + self.dt, velocity)

        preserve_mass = bc["mode"] != "inflow_dirichlet"
        target_mass = None
        if preserve_mass:
            target_mass = np.array([self.total_mass(coeffs_old, col + 1) for col in range(nodal_old.shape[1])], dtype=float)

        rhs_local = nodal_old.copy()
        nodal_current = nodal_old.copy()
        traced_points = trace_points_forward(self.output_nodes, tn, self.dt, velocity, rk)
        rhs_local, traced_points = prepare_forward_boundary_data(self, rhs_local, traced_points, tn, self.dt, nodal_old.shape[1], velocity)

        max_defect_sweeps = 4
        corrections_used = 0
        for _ in range(max_defect_sweeps):
            forward_values = localized_evaluate(self, nodal_current, traced_points)
            residual_local = rhs_local - forward_values
            resid_norm = float(np.max(np.abs(residual_local))) if residual_local.size else 0.0
            mass_resid_norm = 0.0
            mass_resid = None
            if preserve_mass and target_mass is not None:
                current_mass = np.array([self.total_mass(nodal_current, col + 1) for col in range(nodal_current.shape[1])], dtype=float)
                mass_resid = target_mass - current_mass
                mass_resid_norm = float(np.max(np.abs(mass_resid))) if mass_resid.size else 0.0
            if resid_norm <= 1.0e-10 and (not preserve_mass or mass_resid_norm <= 1.0e-12):
                break
            nodal_current += residual_local
            corrections_used += 1
            self.solve_stats["defect_correction_solves"] += 1

        if corrections_used == 0:
            self.solve_stats["moved_solves_zero_defect"] += 1
        elif corrections_used == 1:
            self.solve_stats["moved_solves_one_defect"] += 1
        else:
            self.solve_stats["moved_solves_two_defect"] += 1

        coeffs_new = self.project_samples(nodal_current)
        if preserve_mass and target_mass is not None:
            coeffs_new = enforce_global_mass_correction(self, coeffs_new, target_mass)
        return coeffs_new

    def set_step_size(self, dlt: float) -> None:
        self.dt = dlt

    def reset_solve_stats(self) -> None:
        self.solve_stats = {
            "moved_solves_zero_defect": 0,
            "moved_solves_one_defect": 0,
            "moved_solves_two_defect": 0,
            "defect_correction_solves": 0,
        }

    def get_solve_stats(self) -> dict[str, object]:
        return dict(self.solve_stats)

    def get_num_patches(self) -> int:
        return self.patch_centers.shape[0]

    def get_num_dofs(self) -> int:
        return self.output_nodes.shape[0]

    def get_patch_radius(self) -> float:
        return self.patch_radius

    def get_patch_spacing(self) -> float:
        return self.patch_spacing

    def total_mass(self, coeffs: np.ndarray, col: int = 1) -> float:
        values = np.asarray(coeffs, dtype=float)
        if values.ndim == 2:
            values = values[:, col - 1]
        return float(self.domain_measure * np.mean(values))

    def get_domain_measure(self) -> float:
        return self.domain_measure


def localized_evaluate(obj: PUSLAdvectionSolver, coeffs: np.ndarray, xq: np.ndarray) -> np.ndarray:
    xq = np.asarray(xq, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]
    if xq.size == 0:
        return np.zeros((0, coeffs.shape[1]))

    nq = xq.shape[0]
    nc = coeffs.shape[1]
    values = np.zeros((nq, nc))
    weight_sum = np.zeros(nq)
    patch_ids_per_query = query_patch_ids_for_advection(obj.patch_center_tree, obj.patch_centers, xq, obj.patch_radius)
    for q in range(nq):
        patch_ids = patch_ids_per_query[q]
        if patch_ids.size == 0:
            continue
        center_dist = np.linalg.norm(obj.patch_centers[patch_ids] - xq[q], axis=1)
        alpha = pu_patch_weight(center_dist / obj.patch_radius)
        alpha_sum = float(alpha.sum())
        if alpha_sum <= 1.0e-14:
            alpha = np.ones_like(alpha)
            alpha_sum = float(alpha.sum())
        alpha = alpha / alpha_sum
        for k, p in enumerate(patch_ids):
            ids = obj.patch_node_ids[p]
            floc = coeffs[ids]
            stencil = obj.patch_stencils[p]
            vloc = stencil.eval_weights(obj.patch_stencil_props, xq[q : q + 1]) @ floc
            values[q] += alpha[k] * vloc[0]
        weight_sum[q] = 1.0
    missing = weight_sum <= 1.0e-14
    if np.any(missing):
        idx, _ = obj.domain.query_knn("interior_boundary", xq[missing], 1)
        values[missing] = coeffs[idx[:, 0]]
        weight_sum[missing] = 1.0
    return values / weight_sum[:, None]


def query_patch_ids_for_advection(tree: PatchCenterTree, centers: np.ndarray, xq: np.ndarray, radius: float) -> list[np.ndarray]:
    if tree.has_searcher and tree.searcher is not None:
        patch_ids_per_query = [np.asarray(ids, dtype=int) for ids in tree.searcher.query_ball_point(xq, radius)]
    else:
        d = np.linalg.norm(xq[:, None, :] - centers[None, :, :], axis=2)
        patch_ids_per_query = [np.flatnonzero(d[q] < radius) for q in range(xq.shape[0])]
    for q, patch_ids in enumerate(patch_ids_per_query):
        if patch_ids.size == 0 and centers.size != 0:
            d = np.linalg.norm(centers - xq[q], axis=1)
            patch_ids_per_query[q] = np.array([int(np.argmin(d))], dtype=int)
    return patch_ids_per_query


def trace_points_backward(
    x: np.ndarray,
    tn: float,
    dt: float,
    velocity: Callable[[float, np.ndarray], np.ndarray],
    rk: Callable[[float, np.ndarray, float, Callable[[float, np.ndarray], np.ndarray]], np.ndarray] | None,
) -> np.ndarray:
    return trace_points_signed(x, tn, -dt, velocity, rk)


def trace_points_forward(
    x: np.ndarray,
    tn: float,
    dt: float,
    velocity: Callable[[float, np.ndarray], np.ndarray],
    rk: Callable[[float, np.ndarray, float, Callable[[float, np.ndarray], np.ndarray]], np.ndarray] | None,
) -> np.ndarray:
    return trace_points_signed(x, tn, dt, velocity, rk)


def trace_points_signed(
    x: np.ndarray,
    tn: float,
    dt: float,
    velocity: Callable[[float, np.ndarray], np.ndarray],
    rk: Callable[[float, np.ndarray, float, Callable[[float, np.ndarray], np.ndarray]], np.ndarray] | None,
) -> np.ndarray:
    if callable(rk):
        return np.asarray(rk(tn, x, dt, velocity), dtype=float)
    return rk4_step(tn, x, dt, velocity)


def rk4_step(t: float, x: np.ndarray, dt: float, velocity: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    x1 = x
    k1 = velocity(t, x1)
    x2 = x + 0.5 * dt * k1
    k2 = velocity(t + 0.5 * dt, x2)
    x3 = x + 0.5 * dt * k2
    k3 = velocity(t + 0.5 * dt, x3)
    x4 = x + dt * k3
    k4 = velocity(t + dt, x4)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def apply_boundary_condition(
    obj: PUSLAdvectionSolver,
    values: np.ndarray,
    xdep: np.ndarray,
    tnext: float,
    _velocity: Callable[[float, np.ndarray], np.ndarray],
) -> np.ndarray:
    bc = obj.boundary_condition
    mode = bc["mode"]
    if mode in {"unspecified", "tangential"}:
        return values
    if mode == "inflow_dirichlet":
        phi = obj.domain.get_outer_level_set().evaluate(xdep)
        outside = phi > 0
        if np.any(outside):
            inflow = np.asarray(bc["inflow_value"](tnext, xdep[outside]), dtype=float)
            if inflow.ndim == 1:
                inflow = inflow[:, None]
            values[outside] = inflow
        return values
    if mode == "periodic":
        raise ValueError("Periodic PU-SL transport is not yet supported")
    return values


def validate_boundary_condition_configuration(obj: PUSLAdvectionSolver) -> None:
    bc = obj.boundary_condition
    if bc["mode"] == "periodic":
        if not bc["periodic_patches"]:
            raise ValueError("PUSL periodic advection boundary condition requires periodic patch rules")
        raise ValueError(
            "PUSL periodic advection is not yet supported robustly. "
            "The localized PU basis itself still needs true periodic patch support, not just wrapped traces."
        )
    if bc["mode"] == "inflow_dirichlet" and not callable(bc["inflow_value"]):
        raise ValueError("PUSL inflow-Dirichlet boundary condition requires an inflow callback")


def point_is_inside_level_set(obj: PUSLAdvectionSolver, point: np.ndarray, tol: float | None = None) -> bool:
    phi = obj.domain.get_outer_level_set()
    if phi is None:
        raise ValueError("PUSL advection boundary handling requires a domain level set")
    if tol is None:
        tol = max(1.0e-12, 2.0e-2 * obj.domain.get_sep_rad())
    x = np.asarray(point, dtype=float).reshape(1, -1)
    return float(phi.evaluate(x)[0]) <= tol


def boundary_normal_velocity(point: np.ndarray, normal: np.ndarray, time: float, velocity: Callable[[float, np.ndarray], np.ndarray]) -> float:
    x = np.asarray(point, dtype=float).reshape(1, -1)
    u = np.asarray(velocity(time, x), dtype=float)
    if u.shape != x.shape:
        raise ValueError("velocity callback returned the wrong shape during boundary normal velocity evaluation")
    return float(np.dot(u[0], np.asarray(normal, dtype=float).reshape(-1)))


def validate_tangential_boundary_flow(obj: PUSLAdvectionSolver, time: float, velocity: Callable[[float, np.ndarray], np.ndarray]) -> None:
    xb = obj.domain.get_bdry_nodes()
    nr = obj.domain.get_nrmls()
    if xb.size == 0:
        return
    u = np.asarray(velocity(time, xb), dtype=float)
    if u.shape != xb.shape:
        raise ValueError("PUSL tangential-flow check received a velocity field with the wrong shape")
    normal_speed = np.abs(np.sum(u * nr, axis=1))
    if np.max(normal_speed) > float(obj.boundary_condition["normal_velocity_tolerance"]):
        raise ValueError("PUSL advection tangential-flow boundary condition is incompatible with the supplied velocity field")


def inflow_samples(obj: PUSLAdvectionSolver, time: float, points: np.ndarray, n_cols: int) -> np.ndarray:
    values = np.asarray(obj.boundary_condition["inflow_value"](time, np.asarray(points, dtype=float)), dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (points.shape[0], n_cols):
        raise ValueError("PUSL inflow callback returned a matrix with the wrong shape")
    return values


def boundary_hit_on_segment(obj: PUSLAdvectionSolver, inside_point: np.ndarray, outside_point: np.ndarray, max_iterations: int = 40) -> tuple[np.ndarray, float]:
    phi = obj.domain.get_outer_level_set()
    if phi is None:
        raise ValueError("PUSL boundary tracing requires a domain level set")
    inside = np.asarray(inside_point, dtype=float).reshape(-1)
    outside = np.asarray(outside_point, dtype=float).reshape(-1)
    lo = 0.0
    hi = 1.0
    xlo = inside
    xhi = outside
    flo = float(phi.evaluate(xlo.reshape(1, -1))[0])
    fhi = float(phi.evaluate(xhi.reshape(1, -1))[0])
    tol = max(1.0e-12, 2.0e-2 * obj.domain.get_sep_rad())
    if flo > tol:
        inside, outside = outside, inside
        xlo, xhi = inside, outside
        flo, fhi = fhi, flo
    if flo > tol or fhi < -tol:
        raise ValueError("PUSL boundary tracing failed to bracket a boundary hit on the level set")
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        xmid = (1.0 - mid) * inside + mid * outside
        fmid = float(phi.evaluate(xmid.reshape(1, -1))[0])
        if abs(fmid) <= tol or abs(hi - lo) <= 1.0e-12:
            return xmid, mid
        if fmid <= tol:
            lo = mid
            xlo = xmid
            flo = fmid
        else:
            hi = mid
            xhi = xmid
            fhi = fmid
    mid = 0.5 * (lo + hi)
    return (1.0 - mid) * inside + mid * outside, mid


def prepare_forward_boundary_data(
    obj: PUSLAdvectionSolver,
    rhs_local: np.ndarray,
    traced_points: np.ndarray,
    tn: float,
    dt: float,
    n_cols: int,
    velocity: Callable[[float, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    bc = obj.boundary_condition
    out_rhs = np.asarray(rhs_local, dtype=float).copy()
    out_traced = np.asarray(traced_points, dtype=float).copy()
    xb = obj.domain.get_bdry_nodes()
    nr = obj.domain.get_nrmls()
    n_interior = obj.domain.get_num_interior_nodes()

    for local_row in range(out_traced.shape[0]):
        global_row = local_row
        is_boundary_row = global_row >= n_interior
        x = obj.output_nodes[local_row]

        if bc["mode"] == "inflow_dirichlet" and is_boundary_row:
            boundary_idx = global_row - n_interior
            bn = boundary_normal_velocity(xb[boundary_idx], nr[boundary_idx], tn + dt, velocity)
            if bn < -float(bc["normal_velocity_tolerance"]):
                out_traced[local_row] = x
                out_rhs[local_row] = inflow_samples(obj, tn + dt, x.reshape(1, -1), n_cols)[0]
                continue

        if point_is_inside_level_set(obj, out_traced[local_row]):
            continue

        if bc["mode"] == "tangential":
            raise ValueError("PUSL forward SL exited the domain under a tangential-flow boundary condition")
        if bc["mode"] == "unspecified":
            raise ValueError("PUSL forward SL traced outside the domain without an advection boundary condition")
        if bc["mode"] == "inflow_dirichlet":
            hit_point, _ = boundary_hit_on_segment(obj, x, out_traced[local_row])
            out_traced[local_row] = hit_point
    return out_rhs, out_traced


def sample_backward_traced_values(
    obj: PUSLAdvectionSolver,
    coeffs_old: np.ndarray,
    source_points: np.ndarray,
    traced_points: np.ndarray,
    t_arrival: float,
    velocity: Callable[[float, np.ndarray], np.ndarray],
) -> np.ndarray:
    bc = obj.boundary_condition
    coeffs_old = np.asarray(coeffs_old, dtype=float)
    if coeffs_old.ndim == 1:
        coeffs_old = coeffs_old[:, None]
    traced_points = np.asarray(traced_points, dtype=float)
    source_points = np.asarray(source_points, dtype=float)
    values = np.zeros((traced_points.shape[0], coeffs_old.shape[1]), dtype=float)
    inside_rows: list[int] = []

    for i in range(traced_points.shape[0]):
        if point_is_inside_level_set(obj, traced_points[i]):
            inside_rows.append(i)
            continue
        if bc["mode"] == "tangential":
            raise ValueError("PUSL backward SL traced outside the domain under a tangential-flow boundary condition")
        if bc["mode"] == "unspecified":
            raise ValueError("PUSL backward SL traced outside the domain without an advection boundary condition")
        if bc["mode"] == "inflow_dirichlet":
            hit_point, hit_param = boundary_hit_on_segment(obj, source_points[i], traced_points[i])
            hit_time = t_arrival - hit_param * obj.dt
            values[i] = inflow_samples(obj, hit_time, hit_point.reshape(1, -1), coeffs_old.shape[1])[0]

    if inside_rows:
        ids = np.asarray(inside_rows, dtype=int)
        values[ids] = localized_evaluate(obj, coeffs_old, traced_points[ids])
    return values


def enforce_global_mass_correction(obj: PUSLAdvectionSolver, coeffs: np.ndarray, target_mass: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]
    corrected = coeffs.copy()
    measure = max(obj.get_domain_measure(), 1.0e-14)
    for col in range(corrected.shape[1]):
        current_mass = obj.total_mass(corrected, col + 1)
        alpha = (float(target_mass[col]) - current_mass) / measure
        corrected[:, col] += alpha
    return corrected


def estimate_domain_measure(domain: DomainDescriptor) -> float:
    phi = domain.get_outer_level_set()
    if phi is None:
        return float(domain.get_num_interior_nodes() * domain.get_sep_rad() ** domain.get_dim())
    xall = domain.get_int_bdry_nodes()
    xmin = np.min(xall, axis=0) - domain.get_sep_rad()
    xmax = np.max(xall, axis=0) + domain.get_sep_rad()
    dim = xall.shape[1]
    if dim == 2:
        grid = 32
        x = np.linspace(xmin[0], xmax[0], grid)
        y = np.linspace(xmin[1], xmax[1], grid)
        xg, yg = np.meshgrid(x, y, indexing="ij")
        pts = np.column_stack([xg.ravel(), yg.ravel()])
    elif dim == 3:
        grid = 18
        x = np.linspace(xmin[0], xmax[0], grid)
        y = np.linspace(xmin[1], xmax[1], grid)
        z = np.linspace(xmin[2], xmax[2], grid)
        xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
        pts = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
    else:
        return float(domain.get_num_interior_nodes() * domain.get_sep_rad() ** dim)
    vals = phi.evaluate(pts)
    frac = np.mean(vals <= 0.0)
    return float(frac * np.prod(xmax - xmin))
