from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from kernelpack.domain import DomainDescriptor, DualNodeDomainDescriptor
from kernelpack.solvers.pu_sl_advection import PUSLAdvectionSolver

from .detail import IncompressibleEulerBDFBackend


def _make_physical_advection_domain(velocity_domain: DomainDescriptor) -> DomainDescriptor:
    dim = velocity_domain.get_dim()
    domain = DomainDescriptor()
    domain.set_nodes(velocity_domain.get_interior_nodes(), velocity_domain.get_bdry_nodes(), np.zeros((0, dim)))
    domain.set_normals(velocity_domain.get_nrmls())
    domain.set_sep_rad(velocity_domain.get_sep_rad())
    domain.set_outer_level_set(velocity_domain.get_outer_level_set())
    domain.set_boundary_level_sets(velocity_domain.get_boundary_level_sets())
    domain.build_structs()
    return domain


@dataclass
class PUSLIncompressibleEulerSolver:
    domain: DualNodeDomainDescriptor = field(default_factory=DualNodeDomainDescriptor)
    xi_sl: int = 0
    dt: float = 0.0
    num_omp_threads: int = 1
    advection: PUSLAdvectionSolver = field(default_factory=PUSLAdvectionSolver)
    backend: IncompressibleEulerBDFBackend = field(default_factory=IncompressibleEulerBDFBackend)
    advection_domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    physical_nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    global_physical_nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    interior_nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    coeff_nm2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    coeff_nm1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    coeff_n: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    completed_steps_: int = 0

    def clear_advection_boundary_condition(self) -> None:
        self.advection.clear_advection_boundary_condition()

    def set_tangential_flow_boundary(self, normal_velocity_tolerance: float = 1.0e-10) -> None:
        self.advection.set_tangential_flow_boundary(normal_velocity_tolerance)

    def set_inflow_dirichlet_boundary(self, inflow_value: Callable[..., np.ndarray], normal_velocity_tolerance: float = 1.0e-10) -> None:
        self.advection.set_inflow_dirichlet_boundary(inflow_value, normal_velocity_tolerance)

    def init(
        self,
        domain: DualNodeDomainDescriptor,
        xi_sl: int,
        velocity_sp,
        pressure_sp,
        dt: float,
        num_omp_threads: int = 1,
    ) -> None:
        if not (dt > 0):
            raise ValueError("PUSLIncompressibleEulerSolver requires dt > 0")
        self.domain = domain
        self.domain.build_structs()
        self.xi_sl = int(xi_sl)
        self.dt = float(dt)
        self.num_omp_threads = max(1, int(num_omp_threads))

        self.advection_domain = _make_physical_advection_domain(self.domain.get_velocity_domain())
        self.advection.init(self.advection_domain, xi_sl, dt)
        self.backend.init(self.domain, velocity_sp, pressure_sp, dt, self.num_omp_threads)
        self.physical_nodes = self.advection.get_output_nodes()
        self.global_physical_nodes = self.physical_nodes
        self.interior_nodes = self.domain.get_velocity_domain().get_interior_nodes()
        if (
            self.physical_nodes.shape[1] != self.global_physical_nodes.shape[1]
            or self.global_physical_nodes.shape[1] != self.interior_nodes.shape[1]
        ):
            raise ValueError("PUSLIncompressibleEulerSolver advection/backend dimension mismatch")

        self.coeff_nm2 = np.zeros((0, self.physical_nodes.shape[1]))
        self.coeff_nm1 = np.zeros((0, self.physical_nodes.shape[1]))
        self.coeff_n = np.zeros((0, self.physical_nodes.shape[1]))
        self.completed_steps_ = 0

    def set_step_size(self, dt: float) -> None:
        if not (dt > 0):
            raise ValueError("PUSLIncompressibleEulerSolver requires dt > 0")
        self.dt = float(dt)
        self.advection.set_step_size(dt)
        self.backend.set_step_size(dt)

    def set_initial_velocity(self, physical_velocity: np.ndarray, _global_velocity: np.ndarray | None = None) -> None:
        local_velocity = _normalize_local_physical_velocity(self, physical_velocity, "set_initial_velocity")
        self.coeff_nm2 = self.advection.project_samples(local_velocity)
        self.coeff_nm1 = np.zeros((0, local_velocity.shape[1]))
        self.coeff_n = np.zeros((0, local_velocity.shape[1]))
        self.completed_steps_ = 0
        self.backend.set_initial_velocity_owned(local_velocity)

    def set_velocity_history(self, *velocities: np.ndarray) -> None:
        states = [_normalize_local_physical_velocity(self, velocity, "set_velocity_history") for velocity in velocities]
        self.coeff_nm2 = self.advection.project_samples(states[0])
        if len(states) >= 2:
            self.coeff_nm1 = self.advection.project_samples(states[1])
            self.completed_steps_ = 1
        else:
            self.coeff_nm1 = np.zeros((0, states[0].shape[1]))
            self.completed_steps_ = 0
        if len(states) >= 3:
            self.coeff_n = self.advection.project_samples(states[2])
            self.completed_steps_ = 2
        else:
            self.coeff_n = np.zeros((0, states[0].shape[1]))
        self.backend.set_velocity_history_owned(*states)

    def get_interior_nodes(self) -> np.ndarray:
        return self.interior_nodes

    def get_output_nodes(self) -> np.ndarray:
        return self.physical_nodes

    def get_domain(self) -> DualNodeDomainDescriptor:
        return self.domain

    def advection_solver(self) -> PUSLAdvectionSolver:
        return self.advection

    def euler_backend(self) -> IncompressibleEulerBDFBackend:
        return self.backend

    def bdf1_step(self, t_next: float, rk, body_force: Callable[[float, np.ndarray], np.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        if self.coeff_nm2.size == 0:
            raise ValueError("PUSLIncompressibleEulerSolver.bdf1_step requires set_initial_velocity first")
        transported_nm2 = _transport_to_physical(self, t_next - self.dt, self.coeff_nm2, 1, _extrapolated_velocity_callback(self, t_next, 1), rk)
        self.backend.set_velocity_history_owned(_normalize_local_physical_velocity(self, transported_nm2, "bdf1_step"))
        sol = self.backend.bdf1_step(_force_at(t_next, body_force), problem)
        self.coeff_nm1 = self.advection.project_samples(sol["velocity"])
        self.completed_steps_ = 1
        return sol

    def bdf2_step(self, t_next: float, rk, body_force: Callable[[float, np.ndarray], np.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        if self.completed_steps_ < 1 or self.coeff_nm1.size == 0:
            raise ValueError("PUSLIncompressibleEulerSolver.bdf2_step requires one prior step")
        velocity = _extrapolated_velocity_callback(self, t_next, 2)
        transported_nm2 = _transport_to_physical(self, t_next - 2.0 * self.dt, self.coeff_nm2, 2, velocity, rk)
        transported_nm1 = _transport_to_physical(self, t_next - self.dt, self.coeff_nm1, 1, velocity, rk)
        self.backend.set_velocity_history_owned(
            _normalize_local_physical_velocity(self, transported_nm2, "bdf2_step"),
            _normalize_local_physical_velocity(self, transported_nm1, "bdf2_step"),
        )
        sol = self.backend.bdf2_step(_force_at(t_next, body_force), problem)
        self.coeff_n = self.advection.project_samples(sol["velocity"])
        self.completed_steps_ = 2
        return sol

    def bdf3_step(self, t_next: float, rk, body_force: Callable[[float, np.ndarray], np.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        if self.completed_steps_ < 2 or self.coeff_n.size == 0:
            raise ValueError("PUSLIncompressibleEulerSolver.bdf3_step requires two prior steps")
        velocity = _extrapolated_velocity_callback(self, t_next, 3)
        transported_nm2 = _transport_to_physical(self, t_next - 3.0 * self.dt, self.coeff_nm2, 3, velocity, rk)
        transported_nm1 = _transport_to_physical(self, t_next - 2.0 * self.dt, self.coeff_nm1, 2, velocity, rk)
        transported_n = _transport_to_physical(self, t_next - self.dt, self.coeff_n, 1, velocity, rk)
        self.backend.set_velocity_history_owned(
            _normalize_local_physical_velocity(self, transported_nm2, "bdf3_step"),
            _normalize_local_physical_velocity(self, transported_nm1, "bdf3_step"),
            _normalize_local_physical_velocity(self, transported_n, "bdf3_step"),
        )
        sol = self.backend.bdf3_step(_force_at(t_next, body_force), problem)
        self.coeff_nm2 = self.coeff_nm1
        self.coeff_nm1 = self.coeff_n
        self.coeff_n = self.advection.project_samples(sol["velocity"])
        self.completed_steps_ = max(self.completed_steps_, 3)
        return sol

    def current_interior_velocity(self) -> np.ndarray:
        coeffs = _current_coefficients(self)
        if coeffs.size == 0:
            return np.zeros((0, self.interior_nodes.shape[1]))
        return self.advection.evaluate_at_points(coeffs, self.interior_nodes)

    def current_physical_velocity(self) -> np.ndarray:
        coeffs = _current_coefficients(self)
        if coeffs.size == 0:
            return np.zeros((0, self.physical_nodes.shape[1]))
        return self.advection.evaluate_at_points(coeffs, self.physical_nodes)

    def evaluate_current_velocity_at_points(self, x: np.ndarray) -> np.ndarray:
        coeffs = _current_coefficients(self)
        if coeffs.size == 0:
            return np.zeros((np.asarray(x).shape[0], self.domain.get_dim()))
        return self.advection.evaluate_at_points(coeffs, x)


def _normalize_local_physical_velocity(obj: PUSLIncompressibleEulerSolver, velocity: np.ndarray, where: str) -> np.ndarray:
    dim = obj.domain.get_dim()
    velocity = np.asarray(velocity, dtype=float)
    local_sized = velocity.shape[0] == obj.physical_nodes.shape[0]
    global_sized = velocity.shape[0] == obj.global_physical_nodes.shape[0]
    if (not local_sized and not global_sized) or velocity.shape[1] != dim or np.any(~np.isfinite(velocity)):
        raise ValueError(f"PUSLIncompressibleEulerSolver::{where} received an invalid physical velocity")
    return velocity


def _force_at(t: float, body_force: Callable[[float, np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: body_force(t, x)


def _extrapolation_weights(theta: float, order: int) -> np.ndarray:
    if order <= 1:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if order == 2:
        return np.array([theta + 1.0, -theta, 0.0], dtype=float)
    return np.array(
        [
            0.5 * (theta + 1.0) * (theta + 2.0),
            -theta * (theta + 2.0),
            0.5 * theta * (theta + 1.0),
        ],
        dtype=float,
    )


def _extrapolated_velocity_callback(obj: PUSLIncompressibleEulerSolver, t_next: float, order: int) -> Callable[[float, np.ndarray], np.ndarray]:
    coeff0 = obj.coeff_nm2
    coeff1 = obj.coeff_nm1
    coeff2 = obj.coeff_n
    dt = obj.dt

    def velocity(t: float, x: np.ndarray) -> np.ndarray:
        if coeff0.size == 0:
            return np.zeros((np.asarray(x).shape[0], obj.domain.get_dim()))
        if order <= 1 or coeff1.size == 0:
            return obj.advection.evaluate_at_points(coeff0, x)
        theta = (t - (t_next - dt)) / dt
        w = _extrapolation_weights(theta, 2 if coeff2.size == 0 else order)
        if coeff2.size == 0:
            return w[0] * obj.advection.evaluate_at_points(coeff1, x) + w[1] * obj.advection.evaluate_at_points(coeff0, x)
        return (
            w[0] * obj.advection.evaluate_at_points(coeff2, x)
            + w[1] * obj.advection.evaluate_at_points(coeff1, x)
            + w[2] * obj.advection.evaluate_at_points(coeff0, x)
        )

    return velocity


def _transport_to_physical(
    obj: PUSLIncompressibleEulerSolver,
    t_start: float,
    coeff_state: np.ndarray,
    num_steps: int,
    velocity: Callable[[float, np.ndarray], np.ndarray],
    rk,
) -> np.ndarray:
    if num_steps <= 0:
        raise ValueError("PUSLIncompressibleEulerSolver transport requires a positive step count")
    coeffs = np.asarray(coeff_state, dtype=float)
    t = float(t_start)
    for _ in range(num_steps):
        coeffs = obj.advection.backward_sl_step(t, coeffs, velocity, rk)
        t += obj.dt
    return obj.advection.evaluate_at_points(coeffs, obj.physical_nodes)


def _current_coefficients(obj: PUSLIncompressibleEulerSolver) -> np.ndarray:
    if obj.completed_steps_ <= 0:
        return obj.coeff_nm2
    if obj.completed_steps_ == 1:
        return obj.coeff_nm1
    return obj.coeff_n
