from __future__ import annotations

from math import comb
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from kernelpack.domain import DomainDescriptor
from kernelpack.rbffd import FDODiffOp, FDDiffOp, OpProperties, RBFStencil, StencilProperties, WeightedLeastSquaresStencil


def build_stencil_properties(domain: DomainDescriptor, xi: int, theta: int, point_set: str) -> StencilProperties:
    # The solver layer asks for target order `xi` and operator order `theta`;
    # here we translate that request into a concrete stencil size and
    # polynomial degree that the rbffd layer understands.
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


def resolve_stencil_factory(stencil_spec: str | Callable[[], object]) -> Callable[[], object]:
    # Keep the solver-facing API string-based so examples stay concise, but
    # normalize those strings into actual stencil constructors here.
    if callable(stencil_spec):
        return stencil_spec
    name = str(stencil_spec).lower()
    if name in {"rbf", "rbffd", "rbf-fd"}:
        return lambda: RBFStencil()
    if name in {"wls", "weightedleastsquares", "weighted_least_squares"}:
        return lambda: WeightedLeastSquaresStencil()
    raise ValueError(f"unknown stencil backend {stencil_spec}")


def make_assembler(assembler_spec: str, stencil_spec: str | Callable[[], object]) -> FDDiffOp | FDODiffOp:
    factory = resolve_stencil_factory(stencil_spec)
    name = str(assembler_spec).lower()
    if name in {"fd", "fddiffop", "standard"}:
        return FDDiffOp(factory)
    if name in {"fdo", "fdodiffop", "overlapped", "overlap"}:
        return FDODiffOp(factory)
    raise ValueError(f"unknown assembler {assembler_spec}")


def evaluate_node_callback(func: Callable[..., np.ndarray] | np.ndarray | float, x: np.ndarray, label: str) -> np.ndarray:
    # Accept either callbacks or already-materialized arrays/scalars and always
    # return one value per node.
    values = func(x) if callable(func) else func
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.full(x.shape[0], float(values[0]))
    if values.size != x.shape[0]:
        raise ValueError(f"{label} values must match the node count")
    return values


def evaluate_boundary_values(
    func: Callable[..., np.ndarray] | np.ndarray | float,
    neu_coeff: np.ndarray,
    dir_coeff: np.ndarray,
    nr: np.ndarray,
    xb: np.ndarray,
) -> np.ndarray:
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


def build_system_matrix(lap: sparse.spmatrix, bc: sparse.spmatrix, n_cols: int, pure_neumann: bool) -> sparse.csr_matrix:
    # The stationary Poisson system is built by stacking interior equations and
    # boundary rows. Pure-Neumann problems receive one extra nullspace-fixing
    # row/column to remove the constant mode.
    system = sparse.vstack([-lap, bc], format="csr")
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


def build_system_rhs(rhs_target: np.ndarray, rhs_boundary: np.ndarray, pure_neumann: bool) -> np.ndarray:
    rhs = np.concatenate([rhs_target.reshape(-1), rhs_boundary.reshape(-1)])
    if pure_neumann:
        rhs = np.concatenate([rhs, [0.0]])
    return rhs


def build_initial_guess(
    initial_guess: np.ndarray,
    n_targets: int,
    n_cols: int,
    rhs_boundary: np.ndarray,
    pure_neumann: bool,
) -> np.ndarray | None:
    # Permit several convenient guess shapes so higher-level code can pass just
    # the physical state or the full augmented state depending on what it has.
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


def gmres_with_fallback(system: sparse.spmatrix, rhs: np.ndarray, guess: np.ndarray) -> np.ndarray:
    # Prefer GMRES when the caller supplied an initial guess, but fall back to a
    # direct sparse solve if the Krylov iteration stalls or returns junk.
    sol, info = spla.gmres(system, rhs, x0=guess, rtol=1e-10, atol=0.0, restart=None, maxiter=200)
    if info != 0 or np.any(~np.isfinite(sol)):
        sol = spla.spsolve(system, rhs)
    return np.asarray(sol, dtype=float)


def build_ilu_preconditioner(system: sparse.spmatrix) -> spla.LinearOperator | None:
    # Reuse ILU-preconditioned GMRES in the repeated implicit solves. If ILU
    # fails on a particular sparse system, callers can still fall back to a
    # direct sparse solve.
    try:
        system_csc = system.tocsc(copy=True)
        system_csc.sort_indices()
        ilu = spla.spilu(system_csc)
        return spla.LinearOperator(system.shape, ilu.solve)
    except Exception:
        return None


def gmres_with_preconditioner(
    system: sparse.spmatrix,
    rhs: np.ndarray,
    guess: np.ndarray | None,
    preconditioner: spla.LinearOperator | None,
    *,
    rtol: float = 1e-10,
    maxiter: int = 200,
) -> np.ndarray:
    x0 = None if guess is None or np.asarray(guess).size == 0 else np.asarray(guess, dtype=float).reshape(-1)
    sol, info = spla.gmres(system, rhs, x0=x0, M=preconditioner, rtol=rtol, atol=0.0, restart=None, maxiter=maxiter)
    if info != 0 or np.any(~np.isfinite(sol)):
        sol = spla.spsolve(system, rhs)
    return np.asarray(sol, dtype=float)


def validate_physical_state(state: np.ndarray, n: int) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(-1)
    if state.size != n:
        raise ValueError(f"expected a physical state of length {n}")
    return state


def is_fixed_boundary_callback(func: object) -> bool:
    if not callable(func):
        return False
    try:
        return func.__code__.co_argcount == 1
    except AttributeError:
        return False


def evaluate_boundary_coefficient(
    func: Callable[..., np.ndarray] | np.ndarray | float,
    x: np.ndarray,
    t: float | None = None,
) -> np.ndarray:
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


def evaluate_forcing_callback(
    func: Callable[..., np.ndarray] | np.ndarray | float,
    nu: float,
    t: float,
    x: np.ndarray,
) -> np.ndarray:
    # Time-dependent forcing callbacks in the examples use a few different
    # signatures; this adapter keeps the solver tolerant of those variations.
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


def evaluate_transient_boundary_values(
    func: Callable[..., np.ndarray] | np.ndarray | float,
    neu_coeff: np.ndarray,
    dir_coeff: np.ndarray,
    nr: np.ndarray,
    t: float,
    xb: np.ndarray,
) -> np.ndarray:
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


def build_implicit_system(lap: sparse.csr_matrix, bc: sparse.csr_matrix, n_physical: int, lap_scale: float) -> sparse.csr_matrix:
    # For diffusion-style implicit steps, add the identity contribution only on
    # the physical rows and then append boundary rows underneath.
    system = lap_scale * lap
    system = system.tolil()
    system[:n_physical, :n_physical] = system[:n_physical, :n_physical] + sparse.eye(n_physical, format="lil")
    return sparse.vstack([system.tocsr(), bc], format="csr")
