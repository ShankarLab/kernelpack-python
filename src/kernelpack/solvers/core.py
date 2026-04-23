from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from kernelpack.domain import DomainDescriptor
from kernelpack.rbffd import FDODiffOp, FDDiffOp, OpProperties, RBFStencil, StencilProperties, WeightedLeastSquaresStencil


def _build_stencil_properties(domain: DomainDescriptor, xi: int, theta: int, point_set: str) -> StencilProperties:
    dim = domain.get_dim()
    ell = max(xi + theta - 1, 2)
    sp = StencilProperties()
    sp.dim = dim
    sp.ell = ell
    sp.npoly = int(comb(dim + ell, dim))
    sp.n = 2 * sp.npoly + 1
    sp.spline_degree = ell
    if sp.spline_degree % 2 == 0:
        sp.spline_degree -= 1
    sp.spline_degree = max(sp.spline_degree, 5)
    sp.tree_mode = "all"
    sp.point_set = point_set
    return sp


def _resolve_stencil_factory(stencil_spec: str | Callable[[], object]) -> Callable[[], object]:
    if callable(stencil_spec):
        return stencil_spec
    name = str(stencil_spec).lower()
    if name in {"rbf", "rbffd", "rbf-fd"}:
        return lambda: RBFStencil()
    if name in {"wls", "weightedleastsquares", "weighted_least_squares"}:
        return lambda: WeightedLeastSquaresStencil()
    raise ValueError(f"unknown stencil backend {stencil_spec}")


def _make_assembler(assembler_spec: str, stencil_spec: str | Callable[[], object]) -> FDDiffOp | FDODiffOp:
    factory = _resolve_stencil_factory(stencil_spec)
    name = str(assembler_spec).lower()
    if name in {"fd", "fddiffop", "standard"}:
        return FDDiffOp(factory)
    if name in {"fdo", "fdodiffop", "overlapped", "overlap"}:
        return FDODiffOp(factory)
    raise ValueError(f"unknown assembler {assembler_spec}")


def _evaluate_node_callback(func: Callable[..., np.ndarray] | np.ndarray | float, x: np.ndarray, label: str) -> np.ndarray:
    values = func(x) if callable(func) else func
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.full(x.shape[0], float(values[0]))
    if values.size != x.shape[0]:
        raise ValueError(f"{label} values must match the node count")
    return values


def _evaluate_boundary_values(func: Callable[..., np.ndarray] | np.ndarray | float, neu_coeff: np.ndarray, dir_coeff: np.ndarray, nr: np.ndarray, xb: np.ndarray) -> np.ndarray:
    if callable(func):
        try:
            values = func(neu_coeff, dir_coeff, nr, xb)
        except TypeError:
            values = func(xb)
    else:
        values = func
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != xb.shape[0]:
        raise ValueError("boundary values must match the boundary row count")
    return values


def _build_system_matrix(lap: sparse.spmatrix, bc: sparse.spmatrix, n_cols: int, pure_neumann: bool) -> sparse.csr_matrix:
    a = sparse.vstack([-lap, bc], format="csr")
    if pure_neumann:
        ones_col = sparse.csr_matrix(np.ones((a.shape[0], 1)))
        ones_row = sparse.csr_matrix(np.ones((1, n_cols)))
        a = sparse.vstack([sparse.hstack([a, ones_col], format="csr"), sparse.hstack([ones_row, sparse.csr_matrix([[0.0]])], format="csr")], format="csr")
    return a


def _build_system_rhs(rhs_target: np.ndarray, rhs_boundary: np.ndarray, pure_neumann: bool) -> np.ndarray:
    b = np.concatenate([rhs_target.reshape(-1), rhs_boundary.reshape(-1)])
    if pure_neumann:
        b = np.concatenate([b, [0.0]])
    return b


def _build_initial_guess(initial_guess: np.ndarray, n_targets: int, n_cols: int, rhs_boundary: np.ndarray, pure_neumann: bool) -> np.ndarray | None:
    guess = np.asarray(initial_guess, dtype=float).reshape(-1)
    if guess.size == 0:
        return None
    if pure_neumann:
        if guess.size == n_targets:
            return np.concatenate([guess, rhs_boundary.reshape(-1), [0.0]])
        if guess.size == n_cols:
            return np.concatenate([guess, [0.0]])
        if guess.size == n_cols + 1:
            return guess
        raise ValueError(f"pure-Neumann Poisson guess must have length {n_targets}, {n_cols}, or {n_cols + 1}")
    if guess.size == n_targets:
        return np.concatenate([guess, rhs_boundary.reshape(-1)])
    if guess.size == n_cols:
        return guess
    raise ValueError(f"Poisson guess must have length {n_targets} or {n_cols}")


def _gmres_with_fallback(a: sparse.spmatrix, b: np.ndarray, guess: np.ndarray) -> np.ndarray:
    sol, info = spla.gmres(a, b, x0=guess, rtol=1e-10, atol=0.0, restart=None, maxiter=200)
    if info != 0 or np.any(~np.isfinite(sol)):
        sol = spla.spsolve(a, b)
    return np.asarray(sol, dtype=float)


@dataclass
class PoissonSolver:
    lap_assembler: str = "fd"
    bc_assembler: str = "fd"
    lap_stencil: str = "rbf"
    bc_stencil: str = "rbf"
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
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
    last_solve_used_nullspace_: bool = False

    def init(self, domain: DomainDescriptor, xi: int, num_omp_threads: int = 1) -> None:
        self.domain = domain
        self.xi = xi
        self.num_omp_threads = num_omp_threads
        self.domain.build_structs()
        self.x = self.domain.get_int_bdry_nodes()
        self.xb = self.domain.get_bdry_nodes()
        self.nr = self.domain.get_nrmls()
        self.n = self.x.shape[0]
        self.nf = self.domain.get_num_total_nodes()
        self.lap_stencil_properties = _build_stencil_properties(self.domain, self.xi, 2, "interior_boundary")
        self.bc_stencil_properties = _build_stencil_properties(self.domain, self.xi, 1, "boundary")
        if self.num_omp_threads > 1:
            self.lap_op_properties.use_parallel = True
            self.bc_op_properties.use_parallel = True
        lap_assembler = _make_assembler(self.lap_assembler, self.lap_stencil)
        lap_assembler.assemble_op(self.domain, "lap", self.lap_stencil_properties, self.lap_op_properties)
        self.lap = lap_assembler.get_op().tocsr()
        self.bc = sparse.csr_matrix((0, self.nf))
        self.last_solve_used_nullspace_ = False

    def solve(
        self,
        forcing: Callable[..., np.ndarray] | np.ndarray,
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray] | np.ndarray,
        initial_guess: np.ndarray | None = None,
    ) -> dict[str, object]:
        if initial_guess is None:
            initial_guess = np.zeros(0)
        neu_coeff = _evaluate_node_callback(neu_coeff_func, self.xb, "boundary coefficient")
        dir_coeff = _evaluate_node_callback(dir_coeff_func, self.xb, "boundary coefficient")
        bc_assembler = _make_assembler(self.bc_assembler, self.bc_stencil)
        bc_assembler.assemble_op(self.domain, "bc", self.bc_stencil_properties, self.bc_op_properties, neu_coeff=neu_coeff, dir_coeff=dir_coeff)
        self.bc = bc_assembler.get_op().tocsr()
        rhs_target = _evaluate_node_callback(forcing, self.x, "forcing")
        rhs_boundary = _evaluate_boundary_values(bc, neu_coeff, dir_coeff, self.nr, self.xb)
        pure_neumann = np.max(np.abs(dir_coeff)) <= 1e-13
        self.last_solve_used_nullspace_ = pure_neumann
        a = _build_system_matrix(self.lap, self.bc, self.nf, pure_neumann)
        b = _build_system_rhs(rhs_target, rhs_boundary, pure_neumann)
        guess = _build_initial_guess(initial_guess, self.n, self.nf, rhs_boundary, pure_neumann)
        sol = spla.spsolve(a, b) if guess is None else _gmres_with_fallback(a, b, guess)
        if pure_neumann:
            full_state = sol[: self.nf]
            lagrange_multiplier = sol[-1]
        else:
            full_state = sol
            lagrange_multiplier = None
        return {
            "u": np.asarray(full_state[: self.n], dtype=float),
            "full_state": np.asarray(full_state, dtype=float),
            "L": self.lap,
            "BC": self.bc,
            "system_matrix": a,
            "rhs": b,
            "target_rhs": rhs_target,
            "boundary_rhs": rhs_boundary,
            "used_nullspace_augmentation": pure_neumann,
            "lagrange_multiplier": lagrange_multiplier,
        }

    def get_laplacian(self) -> sparse.csr_matrix:
        return self.lap

    def get_bc_op(self) -> sparse.csr_matrix:
        return self.bc

    def last_solve_used_nullspace(self) -> bool:
        return self.last_solve_used_nullspace_


def _validate_physical_state(state: np.ndarray, n: int) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(-1)
    if state.size != n:
        raise ValueError(f"expected a physical state of length {n}")
    return state


def _is_fixed_boundary_callback(f: object) -> bool:
    if not callable(f):
        return False
    try:
        return f.__code__.co_argcount == 1
    except AttributeError:
        return False


def _evaluate_boundary_coefficient(func: Callable[..., np.ndarray] | np.ndarray | float, x: np.ndarray, t: float | None = None) -> np.ndarray:
    if callable(func):
        if t is None:
            values = func(x)
        else:
            try:
                values = func(t, x)
            except TypeError:
                values = func(x)
    else:
        values = func
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.full(x.shape[0], float(values[0]))
    if values.size != x.shape[0]:
        raise ValueError("boundary coefficients must match the boundary row count")
    return values


def _evaluate_forcing_callback(func: Callable[..., np.ndarray] | np.ndarray | float, nu: float, t: float, x: np.ndarray) -> np.ndarray:
    if callable(func):
        try:
            values = func(nu, t, x)
        except TypeError:
            try:
                values = func(t, x)
            except TypeError:
                values = func(x)
    else:
        values = func
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != x.shape[0]:
        raise ValueError("forcing values must match the physical row count")
    return values


def _evaluate_transient_boundary_values(func: Callable[..., np.ndarray] | np.ndarray | float, neu_coeff: np.ndarray, dir_coeff: np.ndarray, nr: np.ndarray, t: float, xb: np.ndarray) -> np.ndarray:
    if callable(func):
        try:
            values = func(neu_coeff, dir_coeff, nr, t, xb)
        except TypeError:
            try:
                values = func(t, xb)
            except TypeError:
                values = func(xb)
    else:
        values = func
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != xb.shape[0]:
        raise ValueError("boundary values must match the boundary row count")
    return values


def _build_implicit_system(lap: sparse.csr_matrix, bc: sparse.csr_matrix, n_physical: int, lap_scale: float) -> sparse.csr_matrix:
    a = lap_scale * lap
    a = a.tolil()
    a[:n_physical, :n_physical] = a[:n_physical, :n_physical] + sparse.eye(n_physical, format="lil")
    return sparse.vstack([a.tocsr(), bc], format="csr")


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
        self.lap_stencil_properties = _build_stencil_properties(self.domain, self.xi, 2, "interior_boundary")
        self.bc_stencil_properties = _build_stencil_properties(self.domain, self.xi, 1, "boundary")
        if self.num_omp_threads > 1:
            self.lap_op_properties.use_parallel = True
            self.bc_op_properties.use_parallel = True
        lap_assembler = _make_assembler(self.lap_assembler, self.lap_stencil)
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
        self.cnm2 = _validate_physical_state(c0, self.n)
        self.cnm1 = np.zeros(0)
        self.cn = np.zeros(0)
        self.completed_steps_ = 0
        self.fixed_bc_operator_ready_ = False
        self.fixed_bc_coefficients_ready_ = False

    def set_state_history(self, *states: np.ndarray) -> None:
        if len(states) == 1:
            self.set_initial_state(states[0])
        elif len(states) == 2:
            self.cnm2 = _validate_physical_state(states[0], self.n)
            self.cnm1 = _validate_physical_state(states[1], self.n)
            self.cn = np.zeros(0)
            self.completed_steps_ = 1
        elif len(states) == 3:
            self.cnm2 = _validate_physical_state(states[0], self.n)
            self.cnm1 = _validate_physical_state(states[1], self.n)
            self.cn = _validate_physical_state(states[2], self.n)
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

    def bdf1_step(self, t: float, forcing: Callable[..., np.ndarray] | np.ndarray, neu_coeff_func: Callable[..., np.ndarray] | np.ndarray, dir_coeff_func: Callable[..., np.ndarray] | np.ndarray, bc: Callable[..., np.ndarray] | np.ndarray) -> np.ndarray:
        if self.cnm2.size == 0:
            raise ValueError("bdf1_step requires set_initial_state first")
        previous = self.current_physical_state()
        rhs_physical = previous + self.dt * _evaluate_forcing_callback(forcing, self.nu, t, self.x)
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -self.nu * self.dt)

    def bdf2_step(self, t: float, forcing: Callable[..., np.ndarray] | np.ndarray, neu_coeff_func: Callable[..., np.ndarray] | np.ndarray, dir_coeff_func: Callable[..., np.ndarray] | np.ndarray, bc: Callable[..., np.ndarray] | np.ndarray) -> np.ndarray:
        if self.completed_steps_ < 1:
            raise ValueError("bdf2_step requires one prior step in the state history")
        rhs_physical = (4 / 3) * self.cnm1 - (1 / 3) * self.cnm2 + (2 / 3) * self.dt * _evaluate_forcing_callback(forcing, self.nu, t, self.x)
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -(2 / 3) * self.nu * self.dt)

    def bdf3_step(self, t: float, forcing: Callable[..., np.ndarray] | np.ndarray, neu_coeff_func: Callable[..., np.ndarray] | np.ndarray, dir_coeff_func: Callable[..., np.ndarray] | np.ndarray, bc: Callable[..., np.ndarray] | np.ndarray) -> np.ndarray:
        if self.completed_steps_ < 2:
            raise ValueError("bdf3_step requires two prior steps in the state history")
        rhs_physical = (18 / 11) * self.cn - (9 / 11) * self.cnm1 + (2 / 11) * self.cnm2 + (6 / 11) * self.dt * _evaluate_forcing_callback(forcing, self.nu, t, self.x)
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -(6 / 11) * self.nu * self.dt)

    def _take_step(self, rhs_physical: np.ndarray, t: float, neu_coeff_func: Callable[..., np.ndarray] | np.ndarray, dir_coeff_func: Callable[..., np.ndarray] | np.ndarray, bc: Callable[..., np.ndarray] | np.ndarray, lap_scale: float) -> np.ndarray:
        neu_coeff, dir_coeff = self._get_boundary_coefficients(t, neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)
        rhs_boundary = _evaluate_transient_boundary_values(bc, neu_coeff, dir_coeff, self.nr, t, self.xb)
        a = _build_implicit_system(self.lap, self.bc, self.n, lap_scale)
        b = np.concatenate([rhs_physical.reshape(-1), rhs_boundary.reshape(-1)])
        sol = spla.spsolve(a, b)
        next_state = np.asarray(sol[: self.n], dtype=float)
        self._push_completed_step(next_state)
        return self.current_physical_state()

    def _get_boundary_coefficients(self, t: float, neu_coeff_func: Callable[..., np.ndarray] | np.ndarray, dir_coeff_func: Callable[..., np.ndarray] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if _is_fixed_boundary_callback(neu_coeff_func) and _is_fixed_boundary_callback(dir_coeff_func):
            if not self.fixed_bc_coefficients_ready_:
                self.cached_neu_coeff_ = _evaluate_boundary_coefficient(neu_coeff_func, self.xb)
                self.cached_dir_coeff_ = _evaluate_boundary_coefficient(dir_coeff_func, self.xb)
                self.fixed_bc_coefficients_ready_ = True
            return self.cached_neu_coeff_, self.cached_dir_coeff_
        return _evaluate_boundary_coefficient(neu_coeff_func, self.xb, t), _evaluate_boundary_coefficient(dir_coeff_func, self.xb, t)

    def _ensure_boundary_operator(self, neu_coeff: np.ndarray, dir_coeff: np.ndarray) -> None:
        if self.fixed_bc_operator_ready_ and self.bc.shape[0] > 0:
            return
        bc_assembler = _make_assembler(self.bc_assembler, self.bc_stencil)
        bc_assembler.assemble_op(self.domain, "bc", self.bc_stencil_properties, self.bc_op_properties, neu_coeff=neu_coeff, dir_coeff=dir_coeff)
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
