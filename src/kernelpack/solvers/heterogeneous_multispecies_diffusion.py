from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from kernelpack.domain import DomainDescriptor

from .diffusion import DiffusionSolver
from .pu_diffusion import PUDiffusionSolver


@dataclass
class HeterogeneousMultiSpeciesDiffusionSolver:
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    dt: float = 0.0
    nus: np.ndarray = field(default_factory=lambda: np.zeros(0))
    num_omp_threads: int = 1
    solvers: list[DiffusionSolver] = field(default_factory=list)
    lap_assembler: str = "fd"
    bc_assembler: str = "fd"
    lap_stencil: str = "rbf"
    bc_stencil: str = "rbf"

    def init(self, domain: DomainDescriptor, xi: int, dlt: float, d_coeffs: np.ndarray, num_omp_threads: int = 1) -> None:
        self.domain = domain
        self.xi = xi
        self.dt = dlt
        self.nus = np.asarray(d_coeffs, dtype=float).reshape(-1)
        self.num_omp_threads = num_omp_threads
        self.solvers = []
        for nu in self.nus:
            solver = DiffusionSolver(
                lap_assembler=self.lap_assembler,
                bc_assembler=self.bc_assembler,
                lap_stencil=self.lap_stencil,
                bc_stencil=self.bc_stencil,
            )
            solver.init(domain, xi, dlt, float(nu), num_omp_threads)
            self.solvers.append(solver)

    def set_step_size(self, dlt: float) -> None:
        self.dt = dlt
        for solver in self.solvers:
            solver.set_step_size(dlt)

    def set_initial_state(self, u0: np.ndarray) -> None:
        u0 = np.asarray(u0, dtype=float)
        if u0.ndim != 2 or u0.shape[1] != self.nus.size or u0.shape[1] != len(self.solvers):
            raise ValueError("HeterogeneousMultiSpeciesDiffusionSolver was not initialized for the requested number of species")
        for species in range(u0.shape[1]):
            self.solvers[species].set_initial_state(u0[:, species])

    def bdf1_step(self, t: float, forcing: Callable[..., np.ndarray], neu_coeff_func: Callable[..., np.ndarray], dir_coeff_func: Callable[..., np.ndarray], bc: Callable[..., np.ndarray]) -> np.ndarray:
        return self._step_columns(t, forcing, neu_coeff_func, dir_coeff_func, bc, "bdf1_step")

    def bdf2_step(self, t: float, forcing: Callable[..., np.ndarray], neu_coeff_func: Callable[..., np.ndarray], dir_coeff_func: Callable[..., np.ndarray], bc: Callable[..., np.ndarray]) -> np.ndarray:
        return self._step_columns(t, forcing, neu_coeff_func, dir_coeff_func, bc, "bdf2_step")

    def bdf3_step(self, t: float, forcing: Callable[..., np.ndarray], neu_coeff_func: Callable[..., np.ndarray], dir_coeff_func: Callable[..., np.ndarray], bc: Callable[..., np.ndarray]) -> np.ndarray:
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

    def get_bc_ops(self):
        return [solver.bc for solver in self.solvers]

    def get_diffusion_coefficients(self) -> np.ndarray:
        return self.nus.copy()

    def _step_columns(
        self,
        t: float,
        forcing: Callable[..., np.ndarray],
        neu_coeff_func: Callable[..., np.ndarray],
        dir_coeff_func: Callable[..., np.ndarray],
        bc: Callable[..., np.ndarray],
        step_name: str,
    ) -> np.ndarray:
        if not self.solvers:
            raise ValueError("HeterogeneousMultiSpeciesDiffusionSolver requires set_initial_state before stepping")
        out = None
        for species, solver in enumerate(self.solvers):
            nu = float(self.nus[species])
            forcing_species = lambda _, time, x, species=species, nu=nu: np.asarray(forcing(species, nu, time, x), dtype=float).reshape(-1)
            neu_species = lambda time, xb, species=species: np.asarray(neu_coeff_func(species, time, xb), dtype=float).reshape(-1)
            dir_species = lambda time, xb, species=species: np.asarray(dir_coeff_func(species, time, xb), dtype=float).reshape(-1)
            bc_species = (
                lambda neu_coeffs, dir_coeffs, nr, time, xb, species=species: np.asarray(bc(species, neu_coeffs, dir_coeffs, nr, time, xb), dtype=float).reshape(-1)
            )
            col = getattr(solver, step_name)(t, forcing_species, neu_species, dir_species, bc_species)
            if out is None:
                out = np.zeros((col.size, self.nus.size))
            out[:, species] = col
        return out if out is not None else np.zeros((0, 0))


@dataclass
class HeterogeneousMultiSpeciesPUDiffusionSolver:
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    dt: float = 0.0
    nus: np.ndarray = field(default_factory=lambda: np.zeros(0))
    num_omp_threads: int = 1
    solvers: list[PUDiffusionSolver] = field(default_factory=list)

    def init(self, domain: DomainDescriptor, xi: int, dlt: float, d_coeffs: np.ndarray, num_omp_threads: int = 1) -> None:
        self.domain = domain
        self.xi = xi
        self.dt = dlt
        self.nus = np.asarray(d_coeffs, dtype=float).reshape(-1)
        self.num_omp_threads = num_omp_threads
        self.solvers = []
        for nu in self.nus:
            solver = PUDiffusionSolver()
            solver.init(domain, xi, dlt, float(nu), num_omp_threads)
            self.solvers.append(solver)

    def set_step_size(self, dlt: float) -> None:
        self.dt = dlt
        for solver in self.solvers:
            solver.set_step_size(dlt)

    def set_initial_state(self, u0: np.ndarray) -> None:
        u0 = np.asarray(u0, dtype=float)
        if u0.ndim != 2 or u0.shape[1] != self.nus.size or u0.shape[1] != len(self.solvers):
            raise ValueError("HeterogeneousMultiSpeciesPUDiffusionSolver was not initialized for the requested number of species")
        for species in range(u0.shape[1]):
            self.solvers[species].set_initial_state(u0[:, species])

    def bdf1_step(self, t: float, forcing: Callable[..., np.ndarray], neu_coeff_func: Callable[..., np.ndarray], dir_coeff_func: Callable[..., np.ndarray], bc: Callable[..., np.ndarray]) -> np.ndarray:
        return self._step_columns(t, forcing, neu_coeff_func, dir_coeff_func, bc, "bdf1_step")

    def bdf2_step(self, t: float, forcing: Callable[..., np.ndarray], neu_coeff_func: Callable[..., np.ndarray], dir_coeff_func: Callable[..., np.ndarray], bc: Callable[..., np.ndarray]) -> np.ndarray:
        return self._step_columns(t, forcing, neu_coeff_func, dir_coeff_func, bc, "bdf2_step")

    def bdf3_step(self, t: float, forcing: Callable[..., np.ndarray], neu_coeff_func: Callable[..., np.ndarray], dir_coeff_func: Callable[..., np.ndarray], bc: Callable[..., np.ndarray]) -> np.ndarray:
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

    def _step_columns(
        self,
        t: float,
        forcing: Callable[..., np.ndarray],
        neu_coeff_func: Callable[..., np.ndarray],
        dir_coeff_func: Callable[..., np.ndarray],
        bc: Callable[..., np.ndarray],
        step_name: str,
    ) -> np.ndarray:
        if not self.solvers:
            raise ValueError("HeterogeneousMultiSpeciesPUDiffusionSolver requires set_initial_state before stepping")
        out = None
        for species, solver in enumerate(self.solvers):
            nu = float(self.nus[species])
            forcing_species = lambda _, time, x, species=species, nu=nu: np.asarray(forcing(species, nu, time, x), dtype=float).reshape(-1)
            neu_species = lambda time, xb, species=species: np.asarray(neu_coeff_func(species, time, xb), dtype=float).reshape(-1)
            dir_species = lambda time, xb, species=species: np.asarray(dir_coeff_func(species, time, xb), dtype=float).reshape(-1)
            bc_species = (
                lambda neu_coeffs, dir_coeffs, nr, time, xb, species=species: np.asarray(bc(species, neu_coeffs, dir_coeffs, nr, time, xb), dtype=float).reshape(-1)
            )
            col = getattr(solver, step_name)(t, forcing_species, neu_species, dir_species, bc_species)
            if out is None:
                out = np.zeros((col.size, self.nus.size))
            out[:, species] = col
        return out if out is not None else np.zeros((0, 0))
