from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from kernelpack.domain import DomainDescriptor
from kernelpack.rbffd import OpProperties, StencilProperties
from ._common import (
    build_implicit_system,
    build_stencil_properties,
    evaluate_boundary_coefficient,
    evaluate_forcing_callback,
    evaluate_transient_boundary_values,
    is_fixed_boundary_callback,
    make_assembler,
    validate_physical_state,
)


@dataclass
class DiffusionSolver:
    lap_assembler: str = "fd"
    bc_assembler: str = "fd"
    lap_stencil: str = "rbf"
    bc_stencil: str = "rbf"
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    dt: float = np.nan
    nu: float = np.nan
    num_omp_threads: int = 1
    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    nr: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    n: int = 0
    nf: int = 0
    lap: sparse.csr_matrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    bc: sparse.csr_matrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    lap_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    bc_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    lap_op_properties: OpProperties = field(default_factory=lambda: OpProperties(decompose=False, store_weights=True, record_stencils=False))
    bc_op_properties: OpProperties = field(default_factory=lambda: OpProperties(decompose=False, store_weights=True, record_stencils=False))
    cnm2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cnm1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cn: np.ndarray = field(default_factory=lambda: np.zeros(0))
    completed_steps_: int = 0
    fixed_bc_operator_ready_: bool = False
    fixed_bc_coefficients_ready_: bool = False
    cached_neu_coeff_: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cached_dir_coeff_: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def init(self, domain: DomainDescriptor, xi: int, dlt: float, d_coeff: float, num_omp_threads: int = 1) -> None:
        self.domain = domain
        self.xi = xi
        self.dt = dlt
        self.nu = d_coeff
        self.num_omp_threads = num_omp_threads
        self.domain.build_structs()
        self.x = self.domain.get_int_bdry_nodes()
        self.xb = self.domain.get_bdry_nodes()
        self.nr = self.domain.get_nrmls()
        self.n = self.x.shape[0]
        self.nf = self.domain.get_num_total_nodes()
        self.lap_stencil_properties = build_stencil_properties(self.domain, self.xi, 2, "interior_boundary")
        self.bc_stencil_properties = build_stencil_properties(self.domain, self.xi, 1, "boundary")
        if self.num_omp_threads > 1:
            self.lap_op_properties.use_parallel = True
            self.bc_op_properties.use_parallel = True
        lap_assembler = make_assembler(self.lap_assembler, self.lap_stencil)
        lap_assembler.assemble_op(self.domain, "lap", self.lap_stencil_properties, self.lap_op_properties)
        self.lap = lap_assembler.get_op().tocsr()
        self.bc = sparse.csr_matrix((0, self.nf))
        self.cnm2 = np.zeros(0)
        self.cnm1 = np.zeros(0)
        self.cn = np.zeros(0)
        self.completed_steps_ = 0
        self.fixed_bc_operator_ready_ = False
        self.fixed_bc_coefficients_ready_ = False
        self.cached_neu_coeff_ = np.zeros(0)
        self.cached_dir_coeff_ = np.zeros(0)

    def set_step_size(self, dlt: float) -> None:
        self.dt = dlt
        self.fixed_bc_operator_ready_ = False
        self.fixed_bc_coefficients_ready_ = False

    def set_initial_state(self, c0: np.ndarray) -> None:
        self.cnm2 = validate_physical_state(c0, self.n)
        self.cnm1 = np.zeros(0)
        self.cn = np.zeros(0)
        self.completed_steps_ = 0
        self.fixed_bc_operator_ready_ = False
        self.fixed_bc_coefficients_ready_ = False

    def set_state_history(self, *states: np.ndarray) -> None:
        if len(states) == 1:
            self.set_initial_state(states[0])
        elif len(states) == 2:
            self.cnm2 = validate_physical_state(states[0], self.n)
            self.cnm1 = validate_physical_state(states[1], self.n)
            self.cn = np.zeros(0)
            self.completed_steps_ = 1
        elif len(states) == 3:
            self.cnm2 = validate_physical_state(states[0], self.n)
            self.cnm1 = validate_physical_state(states[1], self.n)
            self.cn = validate_physical_state(states[2], self.n)
            self.completed_steps_ = 2
        else:
            raise ValueError("set_state_history expects one, two, or three physical states")
        self.fixed_bc_operator_ready_ = False
        self.fixed_bc_coefficients_ready_ = False

    def current_physical_state(self) -> np.ndarray:
        if self.completed_steps_ <= 0:
            return self.cnm2
        if self.completed_steps_ == 1:
            return self.cnm1
        return self.cn

    def bdf1_step(
        self,
        t: float,
        forcing: Callable[..., np.ndarray] | np.ndarray,
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        if self.cnm2.size == 0:
            raise ValueError("bdf1_step requires set_initial_state first")
        previous = self.current_physical_state()
        rhs_physical = previous + self.dt * evaluate_forcing_callback(forcing, self.nu, t, self.x)
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -self.nu * self.dt)

    def bdf2_step(
        self,
        t: float,
        forcing: Callable[..., np.ndarray] | np.ndarray,
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        if self.completed_steps_ < 1:
            raise ValueError("bdf2_step requires one prior step in the state history")
        rhs_physical = (4 / 3) * self.cnm1 - (1 / 3) * self.cnm2 + (2 / 3) * self.dt * evaluate_forcing_callback(forcing, self.nu, t, self.x)
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -(2 / 3) * self.nu * self.dt)

    def bdf3_step(
        self,
        t: float,
        forcing: Callable[..., np.ndarray] | np.ndarray,
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        if self.completed_steps_ < 2:
            raise ValueError("bdf3_step requires two prior steps in the state history")
        rhs_physical = (18 / 11) * self.cn - (9 / 11) * self.cnm1 + (2 / 11) * self.cnm2 + (6 / 11) * self.dt * evaluate_forcing_callback(forcing, self.nu, t, self.x)
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -(6 / 11) * self.nu * self.dt)

    def _take_step(
        self,
        rhs_physical: np.ndarray,
        t: float,
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray] | np.ndarray,
        lap_scale: float,
    ) -> np.ndarray:
        neu_coeff, dir_coeff = self._get_boundary_coefficients(t, neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)
        rhs_boundary = evaluate_transient_boundary_values(bc, neu_coeff, dir_coeff, self.nr, t, self.xb)
        system = build_implicit_system(self.lap, self.bc, self.n, lap_scale)
        rhs = np.concatenate([rhs_physical.reshape(-1), rhs_boundary.reshape(-1)])
        sol = spla.spsolve(system, rhs)
        next_state = np.asarray(sol[: self.n], dtype=float)
        self._push_completed_step(next_state)
        return self.current_physical_state()

    def _get_boundary_coefficients(
        self,
        t: float,
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if is_fixed_boundary_callback(neu_coeff_func) and is_fixed_boundary_callback(dir_coeff_func):
            if not self.fixed_bc_coefficients_ready_:
                self.cached_neu_coeff_ = evaluate_boundary_coefficient(neu_coeff_func, self.xb)
                self.cached_dir_coeff_ = evaluate_boundary_coefficient(dir_coeff_func, self.xb)
                self.fixed_bc_coefficients_ready_ = True
            return self.cached_neu_coeff_, self.cached_dir_coeff_
        return evaluate_boundary_coefficient(neu_coeff_func, self.xb, t), evaluate_boundary_coefficient(dir_coeff_func, self.xb, t)

    def _ensure_boundary_operator(self, neu_coeff: np.ndarray, dir_coeff: np.ndarray) -> None:
        if self.fixed_bc_operator_ready_ and self.bc.shape[0] > 0:
            return
        bc_assembler = make_assembler(self.bc_assembler, self.bc_stencil)
        bc_assembler.assemble_op(
            self.domain,
            "bc",
            self.bc_stencil_properties,
            self.bc_op_properties,
            neu_coeff=neu_coeff,
            dir_coeff=dir_coeff,
        )
        self.bc = bc_assembler.get_op().tocsr()
        if self.fixed_bc_coefficients_ready_:
            self.fixed_bc_operator_ready_ = True

    def _push_completed_step(self, next_state: np.ndarray) -> None:
        if self.completed_steps_ <= 0:
            self.cnm1 = next_state
        elif self.completed_steps_ == 1:
            self.cn = next_state
        else:
            self.cnm2 = self.cnm1
            self.cnm1 = self.cn
            self.cn = next_state
        self.completed_steps_ += 1
