from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from kernelpack.domain import DomainDescriptor
from kernelpack.rbffd import OpProperties, StencilProperties
from ._common import (
    build_stencil_properties,
    evaluate_boundary_values,
    evaluate_node_callback,
    make_assembler,
)
from .variable_poisson import build_variable_pde_operator, build_variable_system_matrix


def _evaluate_state_callback(
    func: Callable[..., np.ndarray] | np.ndarray | float,
    x: np.ndarray,
    u: np.ndarray,
    label: str,
) -> np.ndarray:
    values = func(x, u) if callable(func) else func
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.full(x.shape[0], float(values[0]))
    if values.size != x.shape[0]:
        raise ValueError(f"{label} values must match the node count")
    return values


def _rectangular_diagonal(values: np.ndarray, n_cols: int) -> sparse.csr_matrix:
    row_ids = np.arange(values.size, dtype=int)
    return sparse.csr_matrix((values, (row_ids, row_ids)), shape=(values.size, n_cols))


def _gmres_step(
    jacobian: sparse.csr_matrix,
    rhs: np.ndarray,
    preconditioner: sparse.csr_matrix,
    linear_tol: float,
) -> np.ndarray:
    ilu_operator = None
    try:
        ilu = spla.spilu(preconditioner.tocsc())
        ilu_operator = spla.LinearOperator(preconditioner.shape, ilu.solve)
    except Exception:
        ilu_operator = None

    delta, info = spla.gmres(
        jacobian,
        rhs,
        M=ilu_operator,
        rtol=linear_tol,
        atol=0.0,
        restart=None,
        maxiter=max(jacobian.shape[0], 1),
    )
    if info != 0 or np.any(~np.isfinite(delta)):
        delta = spla.spsolve(jacobian, rhs)
    return np.asarray(delta, dtype=float)


@dataclass
class NonlinearVariablePoissonSolver:
    lap_assembler: str = "fd"
    bc_assembler: str = "fd"
    lap_stencil: str = "rbf"
    bc_stencil: str = "rbf"
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    num_omp_threads: int = 1
    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xf: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    nr: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    n: int = 0
    nf: int = 0
    lap: sparse.csr_matrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    grad: list[sparse.csr_matrix] = field(default_factory=list)
    pde: sparse.csr_matrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    bc: sparse.csr_matrix = field(default_factory=lambda: sparse.csr_matrix((0, 0)))
    lap_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    bc_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    lap_op_properties: OpProperties = field(default_factory=lambda: OpProperties(decompose=False, store_weights=True, record_stencils=False))
    bc_op_properties: OpProperties = field(default_factory=lambda: OpProperties(decompose=False, store_weights=True, record_stencils=False))
    nonlinear_tol: float = 1.0e-9
    linear_tol: float = 1.0e-9
    max_nonlinear_iterations: int = 20
    last_solve_used_nullspace_: bool = False
    last_nonlinear_iterations_: int = 0
    last_residual_norm_: float = 0.0

    def init(self, domain: DomainDescriptor, xi: int, num_omp_threads: int = 1) -> None:
        self.domain = domain
        self.xi = xi
        self.num_omp_threads = num_omp_threads

        self.domain.build_structs()
        self.x = self.domain.get_int_bdry_nodes()
        self.xb = self.domain.get_bdry_nodes()
        self.xf = self.domain.get_all_nodes()
        self.nr = self.domain.get_nrmls()
        self.n = self.x.shape[0]
        self.nf = self.domain.get_num_total_nodes()

        self.lap_stencil_properties = build_stencil_properties(self.domain, self.xi, 2, "interior_boundary")
        self.bc_stencil_properties = build_stencil_properties(self.domain, self.xi, 1, "boundary")
        self.linear_tol = min(self.domain.get_sep_rad() ** (self.lap_stencil_properties.ell + 1), 1.0e-7)

        if self.num_omp_threads > 1:
            self.lap_op_properties.use_parallel = True
            self.bc_op_properties.use_parallel = True

        lap_assembler = make_assembler(self.lap_assembler, self.lap_stencil)
        lap_assembler.assemble_op(self.domain, "lap", self.lap_stencil_properties, self.lap_op_properties)
        self.lap = lap_assembler.get_op().tocsr()

        self.grad = []
        for d in range(self.domain.get_dim()):
            grad_assembler = make_assembler(self.lap_assembler, self.lap_stencil)
            grad_props = OpProperties(
                selectdim=d,
                decompose=self.lap_op_properties.decompose,
                store_weights=self.lap_op_properties.store_weights,
                record_stencils=self.lap_op_properties.record_stencils,
                nosolve=self.lap_op_properties.nosolve,
                overlap_load=self.lap_op_properties.overlap_load,
                use_parallel=self.lap_op_properties.use_parallel,
            )
            grad_assembler.assemble_op(self.domain, "grad", self.lap_stencil_properties, grad_props)
            self.grad.append(grad_assembler.get_op().tocsr())

        self.pde = sparse.csr_matrix((0, self.nf))
        self.bc = sparse.csr_matrix((0, self.nf))
        self.last_solve_used_nullspace_ = False
        self.last_nonlinear_iterations_ = 0
        self.last_residual_norm_ = 0.0

    def set_nonlinear_tolerance(self, tol: float) -> None:
        self.nonlinear_tol = float(tol)

    def set_linear_tolerance(self, tol: float) -> None:
        self.linear_tol = float(tol)

    def set_max_nonlinear_iterations(self, max_it: int) -> None:
        self.max_nonlinear_iterations = max(1, int(max_it))

    def _assemble_boundary_data(
        self,
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        neu_coeff = evaluate_node_callback(neu_coeff_func, self.xb, "boundary coefficient")
        dir_coeff = evaluate_node_callback(dir_coeff_func, self.xb, "boundary coefficient")
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
        boundary_rhs = evaluate_boundary_values(bc, neu_coeff, dir_coeff, self.nr, self.xb)
        pure_neumann = np.max(np.abs(dir_coeff)) <= 1e-13
        self.last_solve_used_nullspace_ = pure_neumann
        return neu_coeff, dir_coeff, boundary_rhs, pure_neumann

    def _build_initial_unknown(self, initial_guess: np.ndarray, pure_neumann: bool) -> np.ndarray:
        guess = np.asarray(initial_guess, dtype=float).reshape(-1)
        if guess.size == 0:
            return np.zeros(self.nf + (1 if pure_neumann else 0), dtype=float)
        if pure_neumann and guess.size == self.nf + 1:
            return guess.copy()
        if guess.size == self.nf:
            return np.concatenate([guess, [0.0]]) if pure_neumann else guess.copy()
        full_guess = np.zeros(self.nf + (1 if pure_neumann else 0), dtype=float)
        if guess.size == self.n:
            full_guess[: self.n] = guess
            return full_guess
        raise ValueError(
            f"Nonlinear variable Poisson guess must have length {self.n}, {self.nf}"
            + (f", or {self.nf + 1}" if pure_neumann else "")
        )

    def _state_dependent_data(
        self,
        state_all: np.ndarray,
        forcing: Callable[..., np.ndarray] | np.ndarray,
        forcing_u: Callable[..., np.ndarray] | np.ndarray,
        coeff: Callable[..., np.ndarray] | np.ndarray,
        coeff_u: Callable[..., np.ndarray] | np.ndarray,
    ) -> dict[str, np.ndarray | sparse.csr_matrix]:
        state_target = state_all[: self.n]
        coeff_all = _evaluate_state_callback(coeff, self.xf, state_all, "coefficient")
        coeff_u_all = _evaluate_state_callback(coeff_u, self.xf, state_all, "coefficient derivative")
        forcing_local = _evaluate_state_callback(forcing, self.x, state_target, "forcing")
        forcing_u_local = _evaluate_state_callback(forcing_u, self.x, state_target, "forcing derivative")
        pde = build_variable_pde_operator(self.lap, self.grad, coeff_all, self.n)
        return {
            "coeff_all": coeff_all,
            "coeff_u_all": coeff_u_all,
            "forcing_local": forcing_local,
            "forcing_u_local": forcing_u_local,
            "pde": pde,
        }

    def _assemble_residual(
        self,
        state_all: np.ndarray,
        lagrange_multiplier: float,
        boundary_rhs: np.ndarray,
        pure_neumann: bool,
        state_data: dict[str, np.ndarray | sparse.csr_matrix],
    ) -> np.ndarray:
        pde = state_data["pde"]
        forcing_local = np.asarray(state_data["forcing_local"], dtype=float)
        target_residual = np.asarray(pde @ state_all, dtype=float).reshape(-1) - forcing_local
        boundary_residual = np.asarray(self.bc @ state_all, dtype=float).reshape(-1) - boundary_rhs
        if pure_neumann:
            target_residual = target_residual + lagrange_multiplier
            boundary_residual = boundary_residual + lagrange_multiplier
            return np.concatenate([target_residual, boundary_residual, [np.sum(state_all)]])
        return np.concatenate([target_residual, boundary_residual])

    def _assemble_exact_jacobian(
        self,
        state_all: np.ndarray,
        pure_neumann: bool,
        state_data: dict[str, np.ndarray | sparse.csr_matrix],
    ) -> sparse.csr_matrix:
        coeff_u_all = np.asarray(state_data["coeff_u_all"], dtype=float)
        forcing_u_local = np.asarray(state_data["forcing_u_local"], dtype=float)
        pde = state_data["pde"]

        lap_u = np.asarray(self.lap @ state_all, dtype=float).reshape(-1)
        jacobian_target = pde.copy().tocsr()
        jacobian_target = jacobian_target - _rectangular_diagonal(lap_u * coeff_u_all[: self.n], self.nf)

        coeff_u_diag = sparse.diags(coeff_u_all, 0, shape=(self.nf, self.nf), format="csr")
        for grad_op in self.grad:
            grad_u = np.asarray(grad_op @ state_all, dtype=float).reshape(-1)
            jacobian_target = jacobian_target - sparse.diags(grad_u, 0, shape=(self.n, self.n), format="csr") @ grad_op @ coeff_u_diag

        jacobian_target = jacobian_target - _rectangular_diagonal(forcing_u_local, self.nf)
        return build_variable_system_matrix(jacobian_target, self.bc, self.nf, pure_neumann)

    def _assemble_preconditioner(
        self,
        pure_neumann: bool,
        state_data: dict[str, np.ndarray | sparse.csr_matrix],
    ) -> sparse.csr_matrix:
        forcing_u_local = np.asarray(state_data["forcing_u_local"], dtype=float)
        pde = state_data["pde"] - _rectangular_diagonal(forcing_u_local, self.nf)
        return build_variable_system_matrix(pde, self.bc, self.nf, pure_neumann)

    def solve(
        self,
        forcing: Callable[..., np.ndarray] | np.ndarray,
        forcing_u: Callable[..., np.ndarray] | np.ndarray,
        coeff: Callable[..., np.ndarray] | np.ndarray,
        coeff_u: Callable[..., np.ndarray] | np.ndarray,
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray] | np.ndarray,
        initial_guess: np.ndarray | None = None,
    ) -> dict[str, object]:
        if initial_guess is None:
            initial_guess = np.zeros(0)

        neu_coeff, dir_coeff, boundary_rhs, pure_neumann = self._assemble_boundary_data(
            neu_coeff_func,
            dir_coeff_func,
            bc,
        )
        unknown = self._build_initial_unknown(np.asarray(initial_guess, dtype=float), pure_neumann)
        converged = False
        state_data: dict[str, np.ndarray | sparse.csr_matrix] | None = None
        residual = np.zeros_like(unknown)

        for iteration in range(self.max_nonlinear_iterations + 1):
            state_all = unknown[: self.nf]
            lagrange_multiplier = float(unknown[-1]) if pure_neumann else 0.0
            state_data = self._state_dependent_data(state_all, forcing, forcing_u, coeff, coeff_u)
            self.pde = state_data["pde"]
            residual = self._assemble_residual(state_all, lagrange_multiplier, boundary_rhs, pure_neumann, state_data)
            residual_norm = float(np.linalg.norm(residual))
            self.last_nonlinear_iterations_ = iteration
            self.last_residual_norm_ = residual_norm
            if residual_norm <= self.nonlinear_tol:
                converged = True
                break
            if iteration == self.max_nonlinear_iterations:
                break

            jacobian = self._assemble_exact_jacobian(state_all, pure_neumann, state_data)
            preconditioner = self._assemble_preconditioner(pure_neumann, state_data)
            delta = _gmres_step(jacobian, -residual, preconditioner, self.linear_tol)

            step = 1.0
            accepted = False
            for _ in range(8):
                candidate = unknown + step * delta
                candidate_state = candidate[: self.nf]
                candidate_lambda = float(candidate[-1]) if pure_neumann else 0.0
                candidate_data = self._state_dependent_data(candidate_state, forcing, forcing_u, coeff, coeff_u)
                candidate_residual = self._assemble_residual(
                    candidate_state,
                    candidate_lambda,
                    boundary_rhs,
                    pure_neumann,
                    candidate_data,
                )
                if np.linalg.norm(candidate_residual) < residual_norm:
                    unknown = candidate
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                unknown = unknown + delta

        if not converged:
            raise RuntimeError(
                "NonlinearVariablePoissonSolver failed to converge within "
                f"{self.max_nonlinear_iterations} iterations; residual norm = {self.last_residual_norm_:.3e}"
            )

        full_state = np.asarray(unknown[: self.nf], dtype=float)
        lagrange_multiplier = float(unknown[-1]) if pure_neumann else None
        return {
            "u": np.asarray(full_state[: self.n], dtype=float),
            "full_state": full_state,
            "coefficient": np.asarray(state_data["coeff_all"], dtype=float) if state_data is not None else None,
            "coefficient_u": np.asarray(state_data["coeff_u_all"], dtype=float) if state_data is not None else None,
            "L": self.lap,
            "Grad": list(self.grad),
            "PDE": self.pde,
            "BC": self.bc,
            "boundary_rhs": boundary_rhs,
            "neu_coeff": neu_coeff,
            "dir_coeff": dir_coeff,
            "used_nullspace_augmentation": pure_neumann,
            "lagrange_multiplier": lagrange_multiplier,
            "iterations": self.last_nonlinear_iterations_,
            "residual_norm": self.last_residual_norm_,
        }

    def returns_distributed_state(self) -> bool:
        return False

    def get_output_range(self) -> tuple[int, int]:
        return (0, self.n)

    def get_output_nodes(self) -> np.ndarray:
        return self.x

    def get_laplacian(self) -> sparse.csr_matrix:
        return self.lap

    def get_gradient_ops(self) -> list[sparse.csr_matrix]:
        return list(self.grad)

    def get_last_pde_operator(self) -> sparse.csr_matrix:
        return self.pde

    def get_bc_op(self) -> sparse.csr_matrix:
        return self.bc

    def last_solve_used_nullspace(self) -> bool:
        return self.last_solve_used_nullspace_

    def get_last_nonlinear_iterations(self) -> int:
        return self.last_nonlinear_iterations_

    def get_last_residual_norm(self) -> float:
        return self.last_residual_norm_

    def setNonlinearTolerance(self, tol: float) -> None:
        self.set_nonlinear_tolerance(tol)

    def setLinearTolerance(self, tol: float) -> None:
        self.set_linear_tolerance(tol)

    def setMaxNonlinearIterations(self, max_it: int) -> None:
        self.set_max_nonlinear_iterations(max_it)

    def returnsDistributedState(self) -> bool:
        return self.returns_distributed_state()

    def getOutputRange(self) -> tuple[int, int]:
        return self.get_output_range()

    def getOutputNodes(self) -> np.ndarray:
        return self.get_output_nodes()

    def getLaplacian(self) -> sparse.csr_matrix:
        return self.get_laplacian()

    def getGradientOps(self) -> list[sparse.csr_matrix]:
        return self.get_gradient_ops()

    def getLastPdeOperator(self) -> sparse.csr_matrix:
        return self.get_last_pde_operator()

    def getBCOp(self) -> sparse.csr_matrix:
        return self.get_bc_op()

    def lastSolveUsedNullspace(self) -> bool:
        return self.last_solve_used_nullspace()

    def getLastNonlinearIterations(self) -> int:
        return self.get_last_nonlinear_iterations()

    def getLastResidualNorm(self) -> float:
        return self.get_last_residual_norm()
