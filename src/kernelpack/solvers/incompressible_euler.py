from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from kernelpack.domain import DomainDescriptor, DualNodeDomainDescriptor
from kernelpack.rbffd import CrossNodeDiffOp, OpProperties, RBFStencil, StencilProperties
from ._common import build_ilu_preconditioner, gmres_with_preconditioner


def _strip_ghosts(velocity_domain: DomainDescriptor) -> DomainDescriptor:
    dim = velocity_domain.get_dim()
    domain = DomainDescriptor()
    domain.set_nodes(velocity_domain.get_interior_nodes(), velocity_domain.get_bdry_nodes(), np.zeros((0, dim)))
    domain.set_normals(velocity_domain.get_nrmls())
    domain.set_sep_rad(velocity_domain.get_sep_rad())
    domain.set_outer_level_set(velocity_domain.get_outer_level_set())
    domain.set_boundary_level_sets(velocity_domain.get_boundary_level_sets())
    domain.build_structs()
    return domain


def _with_point_set_tree(sp: StencilProperties, point_set: str, tree_mode: str) -> StencilProperties:
    out = StencilProperties(
        n=sp.n,
        dim=sp.dim,
        ell=sp.ell,
        spline_degree=sp.spline_degree,
        npoly=sp.npoly,
        width=sp.width,
        tree_mode=tree_mode,
        point_set=point_set,
    )
    return out


def _boundary_normal_matrix(nr: np.ndarray, n_interior: int, n_phys: int) -> sparse.csr_matrix:
    nub, dim = nr.shape
    rows = []
    cols = []
    vals = []
    for bidx in range(nub):
        boundary_node = n_interior + bidx
        for d in range(dim):
            rows.append(bidx)
            cols.append(d * n_phys + boundary_node)
            vals.append(float(nr[bidx, d]))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(nub, dim * n_phys))


def _boundary_lambda_matrix(nr: np.ndarray, n_interior: int, n_phys: int) -> sparse.csr_matrix:
    nub, dim = nr.shape
    rows = []
    cols = []
    vals = []
    for bidx in range(nub):
        boundary_node = n_interior + bidx
        for d in range(dim):
            rows.append(d * n_phys + boundary_node)
            cols.append(bidx)
            vals.append(float(nr[bidx, d]))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(dim * n_phys, nub))


def _block_divergence_matrix(div_ops: list[sparse.csr_matrix]) -> sparse.csr_matrix:
    return sparse.hstack(div_ops, format="csr")


def _block_gradient_matrix(grad_ops: list[sparse.csr_matrix], scale: float) -> sparse.csr_matrix:
    return sparse.vstack([scale * op for op in grad_ops], format="csr")


@dataclass
class IncompressibleEulerSolver:
    Domain: DualNodeDomainDescriptor = field(default_factory=DualNodeDomainDescriptor)
    VelocityStencilProperties: StencilProperties = field(default_factory=StencilProperties)
    PressureStencilProperties: StencilProperties = field(default_factory=StencilProperties)
    dt: float = 0.0
    num_omp_threads: int = 1
    VelocityPhysicalDomain: DomainDescriptor = field(default_factory=DomainDescriptor)
    Xphys: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Xb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    nr: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Xp: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    GradOps: list[sparse.csr_matrix] = field(default_factory=list)
    DivOps: list[sparse.csr_matrix] = field(default_factory=list)
    velocity_node_range_: tuple[int, int] = (1, 0)
    pressure_node_range_: tuple[int, int] = (1, 0)
    unm2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    unm1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    un: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    completed_steps_: int = 0
    velocity_history_node_owned_: bool = False
    _cached_systems: dict[tuple[float, bool], sparse.csr_matrix] = field(default_factory=dict)
    _cached_preconditioners: dict[tuple[float, bool], spla.LinearOperator | None] = field(default_factory=dict)

    def init(
        self,
        dual_domain: DualNodeDomainDescriptor,
        velocity_sp: StencilProperties,
        pressure_sp: StencilProperties,
        dt: float,
        num_omp_threads: int = 1,
    ) -> None:
        if not (dt > 0):
            raise ValueError("IncompressibleEulerSolver requires dt > 0")
        self.Domain = dual_domain
        self.Domain.build_structs()
        self.VelocityStencilProperties = velocity_sp
        self.PressureStencilProperties = pressure_sp
        self.dt = float(dt)
        self.num_omp_threads = max(1, int(num_omp_threads))

        vel_domain = self.Domain.get_velocity_domain()
        self.VelocityPhysicalDomain = _strip_ghosts(vel_domain)
        self.Xphys = self.VelocityPhysicalDomain.get_int_bdry_nodes()
        self.Xb = self.VelocityPhysicalDomain.get_bdry_nodes()
        self.nr = self.VelocityPhysicalDomain.get_nrmls()
        self.Xp = self.Domain.get_pressure_domain().get_int_bdry_nodes()
        self.velocity_node_range_ = (1, self.Xphys.shape[0])
        self.pressure_node_range_ = (1, self.Xp.shape[0])

        self._assemble_operators()
        self.unm2 = np.zeros((0, self.Xphys.shape[1]))
        self.unm1 = np.zeros((0, self.Xphys.shape[1]))
        self.un = np.zeros((0, self.Xphys.shape[1]))
        self.completed_steps_ = 0
        self.velocity_history_node_owned_ = False
        self._cached_systems.clear()
        self._cached_preconditioners.clear()

    def set_step_size(self, dt: float) -> None:
        if not (dt > 0):
            raise ValueError("IncompressibleEulerSolver requires dt > 0")
        self.dt = float(dt)
        self._cached_systems.clear()
        self._cached_preconditioners.clear()

    def set_initial_velocity(self, u0: np.ndarray) -> None:
        _validate_velocity_state(self, u0, "set_initial_velocity")
        self.velocity_history_node_owned_ = False
        self.unm2 = np.asarray(u0, dtype=float)
        self.unm1 = np.zeros((0, self.unm2.shape[1]))
        self.un = np.zeros((0, self.unm2.shape[1]))
        self.completed_steps_ = 0

    def set_initial_velocity_owned(self, u0_local: np.ndarray) -> None:
        self.set_initial_velocity(u0_local)
        self.velocity_history_node_owned_ = True

    def set_velocity_history(self, *states: np.ndarray) -> None:
        self._set_velocity_history_impl(False, *states)

    def set_velocity_history_owned(self, *states: np.ndarray) -> None:
        self._set_velocity_history_impl(True, *states)

    def get_owned_pressure_range(self) -> tuple[int, int]:
        return self.pressure_node_range_

    def gather_pressure_samples(self, pressure: np.ndarray) -> np.ndarray:
        return np.asarray(pressure, dtype=float)

    def bdf1_step(self, forcing: Callable[[np.ndarray], np.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        if self.unm2.size == 0:
            raise ValueError("IncompressibleEulerSolver.bdf1_step requires set_initial_velocity first")
        return self._solve_bdf_step(1.0, np.array([1.0, 0.0, 0.0]), forcing, problem)

    def bdf2_step(self, forcing: Callable[[np.ndarray], np.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        if self.completed_steps_ < 1:
            raise ValueError("IncompressibleEulerSolver.bdf2_step requires one prior step")
        return self._solve_bdf_step(1.5, np.array([2.0, -0.5, 0.0]), forcing, problem)

    def bdf3_step(self, forcing: Callable[[np.ndarray], np.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        if self.completed_steps_ < 2:
            raise ValueError("IncompressibleEulerSolver.bdf3_step requires two prior steps")
        return self._solve_bdf_step(11.0 / 6.0, np.array([3.0, -1.5, 1.0 / 3.0]), forcing, problem)

    @staticmethod
    def stationary_slip_wall(boundary_indices: np.ndarray) -> dict[str, object]:
        return {
            "boundary_indices": np.asarray(boundary_indices, dtype=int).reshape(-1),
            "normal_velocity": lambda X, nr: np.zeros(X.shape[0]),
        }

    @staticmethod
    def default_problem_definition() -> dict[str, object]:
        return {
            "slip_walls": [],
            "gauge_options": {
                "mode": "automatic",
                "nullspace_tol": 1.0e-7,
                "row_scale_detection": True,
            },
        }

    def _set_velocity_history_impl(self, owned_flag: bool, *states: np.ndarray) -> None:
        for state in states:
            _validate_velocity_state(self, state, "set_velocity_history")
        self.velocity_history_node_owned_ = bool(owned_flag)
        self.unm2 = np.asarray(states[0], dtype=float)
        if len(states) >= 2:
            self.unm1 = np.asarray(states[1], dtype=float)
            self.completed_steps_ = 1
        else:
            self.unm1 = np.zeros((0, self.unm2.shape[1]))
            self.completed_steps_ = 0
        if len(states) >= 3:
            self.un = np.asarray(states[2], dtype=float)
            self.completed_steps_ = 2
        else:
            self.un = np.zeros((0, self.unm2.shape[1]))

    def _assemble_operators(self) -> None:
        dim = self.Domain.get_dim()
        self.GradOps = []
        self.DivOps = []
        op_props = OpProperties(decompose=False, store_weights=True, record_stencils=False)
        for d in range(dim):
            op_props.selectdim = d
            grad_op = CrossNodeDiffOp(lambda: RBFStencil())
            grad_op.assemble_op(
                self.Domain.get_pressure_domain(),
                self.VelocityPhysicalDomain,
                "grad",
                _with_point_set_tree(self.PressureStencilProperties, "interior_boundary", "interior_boundary"),
                op_props,
            )
            self.GradOps.append(grad_op.get_op().tocsr())

            div_op = CrossNodeDiffOp(lambda: RBFStencil())
            div_op.assemble_op(
                self.VelocityPhysicalDomain,
                self.Domain.get_pressure_domain(),
                "grad",
                _with_point_set_tree(self.VelocityStencilProperties, "interior_boundary", "interior_boundary"),
                op_props,
            )
            self.DivOps.append(div_op.get_op().tocsr())

    def _solve_bdf_step(self, alpha0: float, history_coeffs: np.ndarray, forcing: Callable[[np.ndarray], np.ndarray], problem: dict[str, object] | None) -> dict[str, object]:
        problem = _normalize_problem(problem)
        _validate_problem(self, problem)

        dim = self.Domain.get_dim()
        nui = self.VelocityPhysicalDomain.get_num_interior_nodes()
        nub = self.VelocityPhysicalDomain.get_num_bdry_nodes()
        nphys = self.VelocityPhysicalDomain.get_num_int_bdry_nodes()
        npib = self.Domain.get_pressure_domain().get_num_int_bdry_nodes()

        history_rhs = _build_history_rhs(self, history_coeffs, 1.0 / alpha0)
        forcing_u = np.asarray(forcing(self.Xphys), dtype=float)
        if forcing_u.shape != self.Xphys.shape or np.any(~np.isfinite(forcing_u)):
            raise ValueError("IncompressibleEulerSolver forcing callback returned invalid data")
        forcing_u = (self.dt / alpha0) * forcing_u
        prepared = _prepare_wall_data(self, problem)
        boundary_normal_velocity = prepared["boundary_normal_velocity"]

        add_gauge = _pressure_constant_is_null(self, problem["gauge_options"])
        system, preconditioner = self._get_cached_system(alpha0, add_gauge)

        velocity_rhs = (forcing_u + history_rhs).T.reshape(-1)
        rhs = np.concatenate([velocity_rhs, np.zeros(npib, dtype=float), boundary_normal_velocity.reshape(-1)])
        if add_gauge:
            rhs = np.concatenate([rhs, [0.0]])

        guess = self._initial_guess(nphys, npib, nub, add_gauge)
        x = gmres_with_preconditioner(system, rhs, guess, preconditioner)
        if np.any(~np.isfinite(x)):
            raise ValueError("IncompressibleEulerSolver solve returned non-finite values")

        velocity = np.column_stack([x[d * nphys : (d + 1) * nphys] for d in range(dim)])
        pressure = np.asarray(x[dim * nphys : dim * nphys + npib], dtype=float)

        if self.completed_steps_ <= 0:
            self.unm1 = velocity
            self.completed_steps_ = 1
        elif self.completed_steps_ == 1:
            self.un = velocity
            self.completed_steps_ = 2
        else:
            self.unm2 = self.unm1
            self.unm1 = self.un
            self.un = velocity
            self.completed_steps_ = max(self.completed_steps_, 3)

        div_rms, div_max = _divergence_diagnostics(self, velocity)
        wall_rms, wall_max = _wall_diagnostics(self, velocity, boundary_normal_velocity)
        return {
            "velocity": velocity,
            "pressure": pressure,
            "pressure_is_local": False,
            "pressure_begin": 1,
            "pressure_end": npib,
            "divergence_rms": div_rms,
            "divergence_max": div_max,
            "wall_normal_rms": wall_rms,
            "wall_normal_max": wall_max,
        }

    def _get_cached_system(self, alpha0: float, add_gauge: bool) -> tuple[sparse.csr_matrix, spla.LinearOperator | None]:
        key = (float(alpha0), bool(add_gauge))
        if key in self._cached_systems:
            return self._cached_systems[key], self._cached_preconditioners[key]

        dim = self.Domain.get_dim()
        nui = self.VelocityPhysicalDomain.get_num_interior_nodes()
        nub = self.VelocityPhysicalDomain.get_num_bdry_nodes()
        nphys = self.VelocityPhysicalDomain.get_num_int_bdry_nodes()
        npib = self.Domain.get_pressure_domain().get_num_int_bdry_nodes()

        vel_eye = sparse.eye(dim * nphys, format="csr")
        grad_block = _block_gradient_matrix(self.GradOps, self.dt / alpha0)
        div_block = _block_divergence_matrix(self.DivOps)
        lambda_block = _boundary_lambda_matrix(self.nr, nui, nphys)
        wall_block = _boundary_normal_matrix(self.nr, nui, nphys)

        zero_pp = sparse.csr_matrix((npib, npib))
        zero_pl = sparse.csr_matrix((npib, nub))
        zero_lp = sparse.csr_matrix((nub, npib))
        zero_ll = sparse.csr_matrix((nub, nub))
        top = sparse.hstack([vel_eye, grad_block, lambda_block], format="csr")
        middle = sparse.hstack([div_block, zero_pp, zero_pl], format="csr")
        bottom = sparse.hstack([wall_block, zero_lp, zero_ll], format="csr")
        system = sparse.vstack([top, middle, bottom], format="csr")
        if add_gauge:
            gauge_row = sparse.hstack(
                [
                    sparse.csr_matrix((1, dim * nphys)),
                    sparse.csr_matrix(np.ones((1, npib))),
                    sparse.csr_matrix((1, nub)),
                ],
                format="csr",
            )
            gauge_col = sparse.csr_matrix((system.shape[0], 1))
            system = sparse.vstack(
                [
                    sparse.hstack([system, gauge_col], format="csr"),
                    sparse.hstack([gauge_row, sparse.csr_matrix([[0.0]])], format="csr"),
                ],
                format="csr",
            )

        self._cached_systems[key] = system
        self._cached_preconditioners[key] = build_ilu_preconditioner(system)
        return system, self._cached_preconditioners[key]

    def _initial_guess(self, nphys: int, npib: int, nub: int, add_gauge: bool) -> np.ndarray:
        dim = self.Domain.get_dim()
        state = self.un if self.completed_steps_ >= 2 and self.un.size else (self.unm1 if self.completed_steps_ >= 1 and self.unm1.size else self.unm2)
        velocity_guess = np.zeros(dim * nphys, dtype=float)
        if state.size:
            velocity_guess = np.asarray(state, dtype=float).T.reshape(-1)
        tail = np.zeros(npib + nub + int(add_gauge), dtype=float)
        return np.concatenate([velocity_guess, tail])


def _validate_velocity_state(obj: IncompressibleEulerSolver, velocity: np.ndarray, caller: str) -> None:
    velocity = np.asarray(velocity, dtype=float)
    expected_rows = obj.VelocityPhysicalDomain.get_num_int_bdry_nodes()
    expected_cols = obj.Domain.get_dim()
    if velocity.shape != (expected_rows, expected_cols) or np.any(~np.isfinite(velocity)):
        raise ValueError(f"IncompressibleEulerSolver::{caller} received an invalid velocity state")


def _normalize_problem(problem: dict[str, object] | None) -> dict[str, object]:
    if not problem:
        return IncompressibleEulerSolver.default_problem_definition()
    out = dict(problem)
    out.setdefault("slip_walls", [])
    out.setdefault("gauge_options", {"mode": "automatic", "nullspace_tol": 1.0e-7, "row_scale_detection": True})
    return out


def _validate_problem(obj: IncompressibleEulerSolver, problem: dict[str, object]) -> None:
    nub = obj.VelocityPhysicalDomain.get_num_bdry_nodes()
    covered = np.zeros(nub, dtype=int)
    for wall in problem["slip_walls"]:
        idx = np.asarray(wall["boundary_indices"], dtype=int).reshape(-1)
        if np.any((idx < 1) | (idx > nub)):
            raise ValueError("IncompressibleEulerSolver boundary index out of range")
        covered[idx - 1] += 1
    if np.any(covered != 1):
        raise ValueError("IncompressibleEulerSolver slip walls must cover each boundary node exactly once")


def _prepare_wall_data(obj: IncompressibleEulerSolver, problem: dict[str, object]) -> dict[str, np.ndarray]:
    Xb = obj.VelocityPhysicalDomain.get_bdry_nodes()
    nr = obj.VelocityPhysicalDomain.get_nrmls()
    nub = Xb.shape[0]
    boundary_normal_velocity = np.zeros(nub, dtype=float)
    for wall in problem["slip_walls"]:
        idx = np.asarray(wall["boundary_indices"], dtype=int).reshape(-1) - 1
        Xloc = Xb[idx, :]
        nloc = nr[idx, :]
        v = np.asarray(wall["normal_velocity"](Xloc, nloc), dtype=float).reshape(-1)
        if v.size != idx.size or np.any(~np.isfinite(v)):
            raise ValueError("IncompressibleEulerSolver slip wall callback returned invalid normal velocity data")
        boundary_normal_velocity[idx] = v
    return {"boundary_normal_velocity": boundary_normal_velocity}


def _build_history_rhs(obj: IncompressibleEulerSolver, coeffs: np.ndarray, scale: float) -> np.ndarray:
    if obj.completed_steps_ <= 0:
        states = [obj.unm2, None, None]
    elif obj.completed_steps_ == 1:
        states = [obj.unm1, obj.unm2, None]
    else:
        states = [obj.un, obj.unm1, obj.unm2]
    history = np.zeros((obj.Xphys.shape[0], obj.Domain.get_dim()), dtype=float)
    for coeff, state in zip(coeffs, states, strict=True):
        if coeff != 0 and state is not None and np.size(state):
            history += float(coeff) * np.asarray(state, dtype=float)
    return scale * history


def _pressure_constant_is_null(obj: IncompressibleEulerSolver, gauge_options: dict[str, object]) -> bool:
    mode = str(gauge_options.get("mode", "automatic")).lower()
    if mode in {"forcepressuremean", "force_pressure_mean"}:
        return True
    if mode == "none":
        return False
    dim = obj.Domain.get_dim()
    nphys = obj.VelocityPhysicalDomain.get_num_int_bdry_nodes()
    npib = obj.Domain.get_pressure_domain().get_num_int_bdry_nodes()
    y = np.zeros(dim * nphys + npib + obj.VelocityPhysicalDomain.get_num_bdry_nodes(), dtype=float)
    ones = np.ones(npib, dtype=float)
    for d in range(dim):
        y[d * nphys : (d + 1) * nphys] = obj.GradOps[d] @ ones
    residual = np.linalg.norm(y) / max(1.0, np.sqrt(npib))
    return residual <= float(gauge_options.get("nullspace_tol", 1.0e-7))


def _divergence_diagnostics(obj: IncompressibleEulerSolver, velocity: np.ndarray) -> tuple[float, float]:
    npib = obj.Domain.get_pressure_domain().get_num_int_bdry_nodes()
    div = np.zeros(npib, dtype=float)
    for d in range(obj.Domain.get_dim()):
        div += obj.DivOps[d] @ velocity[:, d]
    return float(np.linalg.norm(div) / np.sqrt(max(npib, 1))), float(np.max(np.abs(div), initial=0.0))


def _wall_diagnostics(obj: IncompressibleEulerSolver, velocity: np.ndarray, boundary_normal_velocity: np.ndarray) -> tuple[float, float]:
    nui = obj.VelocityPhysicalDomain.get_num_interior_nodes()
    nub = obj.VelocityPhysicalDomain.get_num_bdry_nodes()
    vals = np.einsum("ij,ij->i", obj.nr, velocity[nui : nui + nub, :])
    resid = vals - boundary_normal_velocity.reshape(-1)
    return float(np.linalg.norm(resid) / np.sqrt(max(nub, 1))), float(np.max(np.abs(resid), initial=0.0))
