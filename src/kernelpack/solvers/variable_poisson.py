from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from kernelpack.domain import DomainDescriptor
from kernelpack.rbffd import OpProperties, StencilProperties
from ._common import (
    build_initial_guess,
    build_stencil_properties,
    build_system_rhs,
    evaluate_boundary_values,
    evaluate_node_callback,
    gmres_with_fallback,
    make_assembler,
)


def build_variable_pde_operator(lap: sparse.spmatrix, grad_ops: list[sparse.spmatrix], coeff_all: np.ndarray, n_rows: int) -> sparse.csr_matrix:
    coeff_local = np.asarray(coeff_all[:n_rows], dtype=float)
    pde = -sparse.diags(coeff_local, 0, shape=(n_rows, n_rows), format="csr") @ lap
    for grad in grad_ops:
        grad_coeff = np.asarray(grad @ coeff_all, dtype=float).reshape(-1)
        pde = pde - sparse.diags(grad_coeff, 0, shape=(n_rows, n_rows), format="csr") @ grad
    return pde.tocsr()


def build_variable_system_matrix(pde: sparse.spmatrix, bc: sparse.spmatrix, n_cols: int, pure_neumann: bool) -> sparse.csr_matrix:
    system = sparse.vstack([pde, bc], format="csr")
    if pure_neumann:
        ones_col = sparse.csr_matrix(np.ones((system.shape[0], 1)))
        ones_row = sparse.csr_matrix(np.ones((1, n_cols)))
        system = sparse.vstack(
            [
                sparse.hstack([system, ones_col], format="csr"),
                sparse.hstack([ones_row, sparse.csr_matrix([[0.0]])], format="csr"),
            ],
            format="csr",
        )
    return system


@dataclass
class VariablePoissonSolver:
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
    last_solve_used_nullspace_: bool = False

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

    def solve(
        self,
        forcing: Callable[..., np.ndarray] | np.ndarray,
        coeff: Callable[..., np.ndarray] | np.ndarray,
        neu_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        dir_coeff_func: Callable[..., np.ndarray] | np.ndarray,
        bc: Callable[..., np.ndarray] | np.ndarray,
        initial_guess: np.ndarray | None = None,
    ) -> dict[str, object]:
        if initial_guess is None:
            initial_guess = np.zeros(0)

        coeff_all = evaluate_node_callback(coeff, self.xf, "coefficient")
        if np.any(coeff_all <= 0):
            raise ValueError("VariablePoissonSolver expects a positive scalar coefficient field")

        self.pde = build_variable_pde_operator(self.lap, self.grad, coeff_all, self.n)

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

        rhs_target = evaluate_node_callback(forcing, self.x, "forcing")
        rhs_boundary = evaluate_boundary_values(bc, neu_coeff, dir_coeff, self.nr, self.xb)
        pure_neumann = np.max(np.abs(dir_coeff)) <= 1e-13
        self.last_solve_used_nullspace_ = pure_neumann

        system = build_variable_system_matrix(self.pde, self.bc, self.nf, pure_neumann)
        rhs = build_system_rhs(rhs_target, rhs_boundary, pure_neumann)
        guess = build_initial_guess(np.asarray(initial_guess, dtype=float).reshape(-1), self.n, self.nf, rhs_boundary, pure_neumann)
        sol = spla.spsolve(system, rhs) if guess is None else gmres_with_fallback(system, rhs, guess)

        if pure_neumann:
            full_state = sol[: self.nf]
            lagrange_multiplier = sol[-1]
        else:
            full_state = sol
            lagrange_multiplier = None

        return {
            "u": np.asarray(full_state[: self.n], dtype=float),
            "full_state": np.asarray(full_state, dtype=float),
            "coefficient": coeff_all,
            "L": self.lap,
            "Grad": list(self.grad),
            "PDE": self.pde,
            "BC": self.bc,
            "system_matrix": system,
            "rhs": rhs,
            "target_rhs": rhs_target,
            "boundary_rhs": rhs_boundary,
            "used_nullspace_augmentation": pure_neumann,
            "lagrange_multiplier": lagrange_multiplier,
        }

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
