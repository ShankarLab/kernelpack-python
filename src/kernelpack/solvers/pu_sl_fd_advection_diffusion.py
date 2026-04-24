from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .diffusion import DiffusionSolver
from .pu_sl_advection import PUSLAdvectionSolver


@dataclass
class PUSLFDAdvectionDiffusionSolver:
    direction: str = "backward"
    advection: PUSLAdvectionSolver = field(default_factory=PUSLAdvectionSolver)
    diffusion: DiffusionSolver = field(default_factory=lambda: DiffusionSolver(lap_stencil="rbf", bc_stencil="rbf"))
    dt: float = 0.0
    nu: float = 0.0
    state_nm2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    state_nm1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    state_n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    completed_steps: int = 0
    enforce_mass_constraint: bool = False
    mass_constraint_target: float = 0.0
    has_explicit_mass_constraint_target: bool = False

    def init(self, domain, xi_sl: int, xi_fd: int, dt: float, nu: float, direction: str = "backward", num_omp_threads: int = 1) -> None:
        self.dt = dt
        self.nu = nu
        self.direction = str(direction).lower()
        self.advection.init(domain, xi_sl, dt)
        self.diffusion.init(domain, xi_fd, dt, nu, num_omp_threads)
        self.state_nm2 = np.zeros(0)
        self.state_nm1 = np.zeros(0)
        self.state_n = np.zeros(0)
        self.completed_steps = 0

    def set_step_size(self, dt: float) -> None:
        self.dt = dt
        self.advection.set_step_size(dt)
        self.diffusion.set_step_size(dt)

    def set_transport_direction(self, direction: str) -> None:
        self.direction = str(direction).lower()

    def enable_homogeneous_neumann_mass_conservation(self, enable: bool = True) -> None:
        self.enable_mass_constraint(enable)

    def enable_mass_constraint(self, enable: bool = True) -> None:
        self.enforce_mass_constraint = bool(enable)
        if enable and not self.has_explicit_mass_constraint_target:
            ref = self.current_state()
            if ref.size:
                self.mass_constraint_target = self.total_mass(ref)

    def disable_mass_constraint(self) -> None:
        self.enforce_mass_constraint = False

    def set_mass_constraint_target(self, target_mass: float) -> None:
        self.mass_constraint_target = float(target_mass)
        self.has_explicit_mass_constraint_target = True
        self.enforce_mass_constraint = True

    def set_initial_state(self, state: np.ndarray) -> None:
        self.state_nm2 = np.asarray(state, dtype=float).reshape(-1)
        self.state_nm1 = np.zeros(0)
        self.state_n = np.zeros(0)
        self.completed_steps = 0
        self._initialize_mass_constraint_target(self.state_nm2)

    def set_state_history(self, *states: np.ndarray) -> None:
        if len(states) == 1:
            self.set_initial_state(states[0])
        elif len(states) == 2:
            self.state_nm2 = np.asarray(states[0], dtype=float).reshape(-1)
            self.state_nm1 = np.asarray(states[1], dtype=float).reshape(-1)
            self.state_n = np.zeros(0)
            self.completed_steps = 1
            self._initialize_mass_constraint_target(self.state_nm1)
        elif len(states) == 3:
            self.state_nm2 = np.asarray(states[0], dtype=float).reshape(-1)
            self.state_nm1 = np.asarray(states[1], dtype=float).reshape(-1)
            self.state_n = np.asarray(states[2], dtype=float).reshape(-1)
            self.completed_steps = 2
            self._initialize_mass_constraint_target(self.state_n)
        else:
            raise ValueError("expected one, two, or three physical states")

    def get_output_nodes(self) -> np.ndarray:
        return self.advection.get_output_nodes()

    def get_output_range(self) -> tuple[int, int]:
        return self.advection.get_output_range()

    def returns_distributed_state(self) -> bool:
        return self.advection.returns_distributed_state()

    def advection_solver(self) -> PUSLAdvectionSolver:
        return self.advection

    def diffusion_solver(self) -> DiffusionSolver:
        return self.diffusion

    def bdf1_step(self, t_next: float, velocity: Callable, rk: Callable | None, forcing: Callable, neu_coeff_func: Callable, dir_coeff_func: Callable, bc: Callable) -> np.ndarray:
        if self.state_nm2.size == 0:
            raise ValueError("bdf1_step requires set_initial_state first")
        transported_nm2 = self._transport_state(t_next - self.dt, self.state_nm2, 1, velocity, rk)
        self.diffusion.set_state_history(transported_nm2)
        next_state = self.diffusion.bdf1_step(t_next, forcing, neu_coeff_func, dir_coeff_func, bc)
        if self.enforce_mass_constraint:
            forcing_mass = self.total_mass(self._evaluate_forcing(forcing, t_next))
            target = self.total_mass(transported_nm2) + self.dt * forcing_mass
            next_state = self._enforce_mass_target(next_state, target)
        self.state_nm1 = next_state
        self.completed_steps = 1
        return next_state

    def bdf2_step(self, t_next: float, velocity: Callable, rk: Callable | None, forcing: Callable, neu_coeff_func: Callable, dir_coeff_func: Callable, bc: Callable) -> np.ndarray:
        if self.completed_steps < 1 or self.state_nm1.size == 0:
            raise ValueError("bdf2_step requires one prior step")
        transported_nm2 = self._transport_state(t_next - 2 * self.dt, self.state_nm2, 2, velocity, rk)
        transported_nm1 = self._transport_state(t_next - self.dt, self.state_nm1, 1, velocity, rk)
        self.diffusion.set_state_history(transported_nm2, transported_nm1)
        next_state = self.diffusion.bdf2_step(t_next, forcing, neu_coeff_func, dir_coeff_func, bc)
        if self.enforce_mass_constraint:
            forcing_mass = self.total_mass(self._evaluate_forcing(forcing, t_next))
            target = (4 * self.total_mass(transported_nm1) - self.total_mass(transported_nm2) + 2 * self.dt * forcing_mass) / 3
            next_state = self._enforce_mass_target(next_state, target)
        self.state_n = next_state
        self.completed_steps = 2
        return next_state

    def bdf3_step(self, t_next: float, velocity: Callable, rk: Callable | None, forcing: Callable, neu_coeff_func: Callable, dir_coeff_func: Callable, bc: Callable) -> np.ndarray:
        if self.completed_steps < 2 or self.state_n.size == 0:
            raise ValueError("bdf3_step requires two prior steps")
        transported_nm2 = self._transport_state(t_next - 3 * self.dt, self.state_nm2, 3, velocity, rk)
        transported_nm1 = self._transport_state(t_next - 2 * self.dt, self.state_nm1, 2, velocity, rk)
        transported_n = self._transport_state(t_next - self.dt, self.state_n, 1, velocity, rk)
        self.diffusion.set_state_history(transported_nm2, transported_nm1, transported_n)
        next_state = self.diffusion.bdf3_step(t_next, forcing, neu_coeff_func, dir_coeff_func, bc)
        if self.enforce_mass_constraint:
            forcing_mass = self.total_mass(self._evaluate_forcing(forcing, t_next))
            target = (18 * self.total_mass(transported_n) - 9 * self.total_mass(transported_nm1) + 2 * self.total_mass(transported_nm2) + 6 * self.dt * forcing_mass) / 11
            next_state = self._enforce_mass_target(next_state, target)
        self.state_nm2 = self.state_nm1
        self.state_nm1 = self.state_n
        self.state_n = next_state
        self.completed_steps = max(self.completed_steps, 3)
        return next_state

    def current_state(self) -> np.ndarray:
        if self.completed_steps <= 0:
            return self.state_nm2
        if self.completed_steps == 1:
            return self.state_nm1
        return self.state_n

    def total_mass(self, local_state: np.ndarray) -> float:
        return self.advection.total_mass(np.asarray(local_state, dtype=float).reshape(-1), 1)

    def _transport_state(self, t_start: float, state: np.ndarray, num_steps: int, velocity: Callable, rk: Callable | None) -> np.ndarray:
        original_dt = self.dt
        horizon = num_steps * self.dt
        self.advection.set_step_size(horizon)
        if self.direction == "forward":
            coeffs_next = self.advection.forward_sl_step(t_start, np.asarray(state, dtype=float).reshape(-1), velocity, rk)
        else:
            coeffs_next = self.advection.backward_sl_step(t_start, np.asarray(state, dtype=float).reshape(-1), velocity, rk)
        self.advection.set_step_size(original_dt)
        transported = np.asarray(coeffs_next, dtype=float).reshape(-1)
        if self.enforce_mass_constraint:
            transported = self._enforce_mass_target(transported, self.mass_constraint_target)
        return transported

    def _evaluate_forcing(self, forcing: Callable, t: float) -> np.ndarray:
        x = self.advection.get_output_nodes()
        try:
            values = forcing(self.nu, t, x)
        except TypeError:
            try:
                values = forcing(t, x)
            except TypeError:
                values = forcing(x)
        return np.asarray(values, dtype=float).reshape(-1)

    def _initialize_mass_constraint_target(self, reference_state: np.ndarray) -> None:
        if self.enforce_mass_constraint and not self.has_explicit_mass_constraint_target:
            self.mass_constraint_target = self.total_mass(reference_state)

    def _enforce_mass_target(self, state: np.ndarray, target_mass: float) -> np.ndarray:
        current_mass = self.total_mass(state)
        alpha = (target_mass - current_mass) / max(self.advection.get_domain_measure(), 1.0e-14)
        return np.asarray(state, dtype=float) + alpha
