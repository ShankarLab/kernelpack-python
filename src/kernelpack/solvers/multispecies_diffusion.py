from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from kernelpack.domain import DomainDescriptor

from .diffusion import DiffusionSolver


@dataclass
class MultiSpeciesDiffusionSolver:
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    dt: float = 0.0
    nu: float = 0.0
    num_omp_threads: int = 1
    solvers: list[DiffusionSolver] = field(default_factory=list)
    num_species: int = 0
    lap_assembler: str = "fd"
    bc_assembler: str = "fd"
    lap_stencil: str = "rbf"
    bc_stencil: str = "rbf"

    def init(self, domain: DomainDescriptor, xi: int, dlt: float, d_coeff: float, num_omp_threads: int = 1) -> None:
        self.domain = domain
        self.xi = xi
        self.dt = dlt
        self.nu = d_coeff
        self.num_omp_threads = num_omp_threads
        self.solvers = []
        self.num_species = 0

    def set_step_size(self, dlt: float) -> None:
        self.dt = dlt
        for solver in self.solvers:
            solver.set_step_size(dlt)

    def set_initial_state(self, u0: np.ndarray) -> None:
        u0 = np.asarray(u0, dtype=float)
        if u0.ndim != 2:
            raise ValueError("MultiSpeciesDiffusionSolver expects a 2D state matrix")
        self._ensure_solvers(u0.shape[1])
        for species in range(u0.shape[1]):
            self.solvers[species].set_initial_state(u0[:, species])

    def set_state_history(self, *states: np.ndarray) -> None:
        if not states:
            raise ValueError("set_state_history expects at least one state")
        num_species = np.asarray(states[0], dtype=float).shape[1]
        self._ensure_solvers(num_species)
        for state in states[1:]:
            if np.asarray(state, dtype=float).shape[1] != num_species:
                raise ValueError("all state-history matrices must have the same species count")
        for species in range(num_species):
            cols = [np.asarray(state, dtype=float)[:, species] for state in states]
            self.solvers[species].set_state_history(*cols)

    def bdf1_step(
        self,
        t: float,
        forcing: Callable[..., np.ndarray],
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray],
    ) -> np.ndarray:
        return self._step_columns(t, forcing, neu_coeff_func, dir_coeff_func, bc, "bdf1_step")

    def bdf2_step(
        self,
        t: float,
        forcing: Callable[..., np.ndarray],
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray],
    ) -> np.ndarray:
        return self._step_columns(t, forcing, neu_coeff_func, dir_coeff_func, bc, "bdf2_step")

    def bdf3_step(
        self,
        t: float,
        forcing: Callable[..., np.ndarray],
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray],
    ) -> np.ndarray:
        return self._step_columns(t, forcing, neu_coeff_func, dir_coeff_func, bc, "bdf3_step")

    def returns_distributed_state(self) -> bool:
        return bool(self.solvers) and self.solvers[0].returns_distributed_state()

    def get_output_range(self) -> tuple[int, int]:
        if not self.solvers:
            n = self.domain.get_num_int_bdry_nodes()
            return (1, n)
        return self.solvers[0].get_output_range()

    def get_output_nodes(self) -> np.ndarray:
        if not self.solvers:
            return self.domain.get_int_bdry_nodes()
        return self.solvers[0].get_output_nodes()

    def get_laplacian(self):
        return self.solvers[0].lap if self.solvers else None

    def get_bc_op(self):
        return self.solvers[0].bc if self.solvers else None

    def _ensure_solvers(self, num_species: int) -> None:
        if num_species <= 0:
            raise ValueError("MultiSpeciesDiffusionSolver requires at least one species")
        if not self.solvers:
            self.solvers = []
            for _ in range(num_species):
                solver = DiffusionSolver(
                    lap_assembler=self.lap_assembler,
                    bc_assembler=self.bc_assembler,
                    lap_stencil=self.lap_stencil,
                    bc_stencil=self.bc_stencil,
                )
                solver.init(self.domain, self.xi, self.dt, self.nu, self.num_omp_threads)
                self.solvers.append(solver)
            self.num_species = num_species
            return
        if self.num_species != num_species:
            raise ValueError("MultiSpeciesDiffusionSolver was initialized for a different number of species")

    def _step_columns(
        self,
        t: float,
        forcing: Callable[..., np.ndarray],
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray],
        step_name: str,
    ) -> np.ndarray:
        if not self.solvers:
            raise ValueError("MultiSpeciesDiffusionSolver requires set_initial_state before stepping")
        out = None
        for species in range(self.num_species):
            forcing_species = lambda nu, time, x, species=species: np.asarray(forcing(nu, time, x), dtype=float)[:, species]
            bc_species = (
                lambda neu_coeffs, dir_coeffs, nr, time, xb, species=species: np.asarray(bc(neu_coeffs, dir_coeffs, nr, time, xb), dtype=float)[:, species]
            )
            col = getattr(self.solvers[species], step_name)(t, forcing_species, neu_coeff_func, dir_coeff_func, bc_species)
            if out is None:
                out = np.zeros((col.size, self.num_species))
            out[:, species] = col
        return out if out is not None else np.zeros((0, 0))
