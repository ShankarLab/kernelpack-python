from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil
from typing import Callable

import numpy as np
from scipy import sparse
from scipy import linalg as dense_linalg

from kernelpack._numba import build_augmented_rbf_lhs, normalize_stencil_points, phs_dr_over_r_matrix, phs_kernel_matrix, phs_lap_matrix
from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import distance_matrix
from kernelpack.poly import PolynomialBasis, total_degree_indices


def _unit_multi_index(dim: int, selectdim: int) -> np.ndarray:
    d = np.zeros((1, dim), dtype=int)
    d[0, selectdim] = 1
    return d


@lru_cache(maxsize=32)
def _cached_unit_multi_index(dim: int, selectdim: int) -> np.ndarray:
    return _unit_multi_index(dim, selectdim)


@lru_cache(maxsize=32)
def _zero_derivative(dim: int) -> np.ndarray:
    return np.zeros((1, dim), dtype=int)


@lru_cache(maxsize=128)
def _basis_template(dim: int, ell: int, family: str) -> PolynomialBasis:
    return PolynomialBasis.from_total_degree(dim, ell, family=family, center=np.zeros(dim), scale=1.0)


@dataclass
class StencilProperties:
    n: int = 0
    dim: int = 0
    ell: int = 0
    spline_degree: int = 3
    npoly: int = 0
    width: float = 1.0
    tree_mode: str = "all"
    point_set: str = "interior_boundary"

    def __post_init__(self) -> None:
        self.tree_mode = self.normalize_tree_mode(self.tree_mode)
        self.point_set = self.normalize_point_set(self.point_set)
        if self.npoly == 0 and self.dim > 0:
            self.npoly = total_degree_indices(self.dim, self.ell).shape[0]

    @classmethod
    def from_accuracy(
        cls,
        *,
        operator: str = "lap",
        convergence_order: int | None = None,
        diff_op_order: int | None = None,
        dimension: int,
        approximation: str = "rbf",
        stencil_factor: float | None = None,
        spline_degree: int | None = None,
        tree_mode: str = "all",
        point_set: str = "interior_boundary",
    ) -> "StencilProperties":
        # Translate a user-facing accuracy request into the internal stencil
        # parameters used by the discrete operator builders. The exact formulas
        # here are heuristic but intentionally mirror the Matlab-side defaults.
        q = cls.default_diff_order(operator) if diff_op_order is None else diff_op_order
        p = 2 if convergence_order is None else convergence_order
        ell = max(p + q - 1, 0)
        npoly = total_degree_indices(dimension, ell).shape[0]
        approx = approximation.lower()
        if stencil_factor is None:
            if approx in {"rbf", "rbf-fd", "rbffd"}:
                stencil_factor = 2.0
            elif approx in {"wls", "weighted_least_squares", "weightedleastsquares"}:
                stencil_factor = 1.5
            else:
                raise ValueError(f"unknown approximation {approximation}")
        if spline_degree is None:
            spline_degree = max(2 * q + 1, 3)
        if spline_degree % 2 == 0:
            spline_degree -= 1
        n = max(npoly + 1, ceil(stencil_factor * npoly))
        return cls(n=n, dim=dimension, ell=ell, spline_degree=spline_degree, npoly=npoly, tree_mode=tree_mode, point_set=point_set)

    @staticmethod
    def normalize_tree_mode(mode: str | int) -> str:
        if isinstance(mode, (int, np.integer)):
            return ["all", "interior_boundary", "boundary"][int(mode)]
        mode = str(mode).lower()
        aliases = {
            "all": "all",
            "all_nodes": "all",
            "full": "all",
            "interior_boundary": "interior_boundary",
            "interior+boundary": "interior_boundary",
            "int_bdry": "interior_boundary",
            "intboundary": "interior_boundary",
            "owned": "interior_boundary",
            "boundary": "boundary",
            "bdry": "boundary",
            "boundary_only": "boundary",
        }
        if mode not in aliases:
            raise ValueError(f"unknown tree mode {mode}")
        return aliases[mode]

    @staticmethod
    def normalize_point_set(mode: str | int) -> str:
        if isinstance(mode, (int, np.integer)):
            return ["all", "interior_boundary", "boundary"][int(mode)]
        mode = str(mode).lower()
        aliases = {
            "all": "all",
            "all_nodes": "all",
            "full": "all",
            "interior_boundary": "interior_boundary",
            "interior+boundary": "interior_boundary",
            "int_bdry": "interior_boundary",
            "intboundary": "interior_boundary",
            "interior and boundary": "interior_boundary",
            "boundary": "boundary",
            "bdry": "boundary",
            "boundary_only": "boundary",
        }
        if mode not in aliases:
            raise ValueError(f"unknown point set {mode}")
        return aliases[mode]

    @staticmethod
    def default_diff_order(op: str) -> int:
        op = str(op).lower()
        if op in {"interp", "interpolation", "identity"}:
            return 0
        if op in {"grad", "gradient", "dx", "dy", "dz"}:
            return 1
        if op in {"lap", "laplacian", "bc", "boundary"}:
            return 2
        raise ValueError(f"unknown operator {op}")


@dataclass
class OpProperties:
    selectdim: int = 0
    decompose: bool = True
    store_weights: bool = True
    record_stencils: bool = False
    nosolve: bool = False
    overlap_load: float = 0.3
    use_parallel: bool = False


@dataclass
class RBFStencil:
    a: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    solve_lhs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    x_stencil: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    r_stencil: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xc: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    width: float = 1.0
    s_dim: int = 0
    n: int = 0
    ell: int = 0
    npoly: int = 0
    basis: PolynomialBasis | None = None
    coeffs_already_computed: bool = False
    wt: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    l: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    bc: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    gx: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    gy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    gz: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def initialize_geometry(self, x: np.ndarray, sp: StencilProperties) -> None:
        # Normalize the stencil cloud around its centroid and assemble the
        # augmented RBF-FD interpolation matrix once. Every operator row for
        # this stencil reuses the same left-hand side.
        self.coeffs_already_computed = False
        self.s_dim = x.shape[1]
        self.n = x.shape[0]
        self.x_stencil = x
        self.ell = sp.ell
        self.npoly = sp.npoly
        self.r_stencil = distance_matrix(x, x)
        self.width = max(float(self.r_stencil.max(initial=0.0)), 1.0)
        self.xm, self.xc, self.width = normalize_stencil_points(x, self.width)
        self.basis = _basis_template(self.s_dim, self.ell, "legendre")
        p = self.basis.evaluate(self.xc, _zero_derivative(self.s_dim), True)
        self.a = build_augmented_rbf_lhs(self.phs_rbf(self.r_stencil, sp.spline_degree), p)
        self.solve_lhs = self.a

    def compute_weights(self, x: np.ndarray, *args: object) -> np.ndarray:
        # The Matlab code routes both interior and boundary rows through the
        # same public entry point. We preserve that API shape here and branch
        # based on the argument pattern.
        if len(args) >= 7 and isinstance(args[0], np.ndarray) and args[0].shape[1] == x.shape[1]:
            nr, neu_coeff, dir_coeff, sp, op, apply_op, rhs_indices = args[:7]
            return self._compute_weights_boundary(x, nr, float(neu_coeff), float(dir_coeff), sp, op, apply_op, rhs_indices)
        sp, op, apply_op, rhs_indices = args[:4]
        return self._compute_weights_interior(x, sp, op, apply_op, rhs_indices)

    def eval_weights(self, sp: StencilProperties, xe: np.ndarray) -> np.ndarray:
        xe = np.asarray(xe, dtype=float)
        if xe.size == 0:
            return np.zeros((0, self.n))
        re = distance_matrix(xe, self.x_stencil)
        xec = (xe - self.xm) / self.width
        pe = self.basis.evaluate(xec, np.zeros((1, self.s_dim), dtype=int), True)
        rt = np.vstack([self.phs_rbf(re, sp.spline_degree).T, pe.T])
        lagrange = self.stable_solve(self.solve_lhs, rt)
        return lagrange[: self.n].T

    def get_interp_mat(self) -> np.ndarray:
        return self.a[: self.n, : self.n]

    def lap_op(self, sp: StencilProperties, _op: OpProperties, r_rhs: np.ndarray, _x_subset: np.ndarray, _x: np.ndarray, x_at_origin_subset: np.ndarray, _x_at_origin: np.ndarray) -> np.ndarray:
        bpoly = np.zeros((self.npoly, x_at_origin_subset.shape[0]))
        for d in range(self.s_dim):
            bpoly += self.basis.evaluate(x_at_origin_subset, _cached_unit_multi_index(self.s_dim, d) + _cached_unit_multi_index(self.s_dim, d), True).T / (self.width**2)
        top = self.phs_lap(r_rhs, sp.spline_degree, self.s_dim).T
        out = np.empty((top.shape[0] + bpoly.shape[0], top.shape[1]), dtype=float)
        out[: top.shape[0], :] = top
        out[top.shape[0] :, :] = bpoly
        return out

    def grad_op(self, sp: StencilProperties, op: OpProperties, r_rhs: np.ndarray, x_subset: np.ndarray, x: np.ndarray, x_at_origin_subset: np.ndarray, _x_at_origin: np.ndarray) -> np.ndarray:
        dim = op.selectdim
        diff = x_subset[:, dim : dim + 1] - x[None, :, dim]
        bpoly = self.basis.evaluate(x_at_origin_subset, _cached_unit_multi_index(self.s_dim, dim), True).T / self.width
        top = (diff * self.phs_dr_over_r(r_rhs, sp.spline_degree)).T
        out = np.empty((top.shape[0] + bpoly.shape[0], top.shape[1]), dtype=float)
        out[: top.shape[0], :] = top
        out[top.shape[0] :, :] = bpoly
        return out

    def bc_op(self, sp: StencilProperties, op: OpProperties, neu_coeff: float, dir_coeff: float, r_rhs: np.ndarray, x_subset: np.ndarray, x: np.ndarray, x_at_origin_subset: np.ndarray, x_at_origin: np.ndarray, nr_subset: np.ndarray) -> np.ndarray:
        # Mixed boundary rows are assembled as a linear combination of normal
        # derivative and interpolation rows. This keeps the row construction
        # close to the mathematical boundary condition.
        total = np.zeros((self.n + self.npoly, x_at_origin_subset.shape[0]))
        if neu_coeff != 0:
            for d in range(self.s_dim):
                diff = x_subset[:, d : d + 1] - x[None, :, d]
                grad_rbf = (diff * self.phs_dr_over_r(r_rhs, sp.spline_degree)).T
                grad_poly = self.basis.evaluate(x_at_origin_subset, _cached_unit_multi_index(self.s_dim, d), True).T / self.width
                stacked = np.empty((grad_rbf.shape[0] + grad_poly.shape[0], grad_rbf.shape[1]), dtype=float)
                stacked[: grad_rbf.shape[0], :] = grad_rbf
                stacked[grad_rbf.shape[0] :, :] = grad_poly
                total += neu_coeff * stacked * nr_subset[:, d]
        if dir_coeff != 0:
            total += dir_coeff * self.interp_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        if dir_coeff == 0 and neu_coeff == 0:
            raise ValueError("both boundary coefficients cannot be zero")
        return total

    def interp_op(self, sp: StencilProperties, _op: OpProperties, r_rhs: np.ndarray, _x_subset: np.ndarray, _x: np.ndarray, x_at_origin_subset: np.ndarray, _x_at_origin: np.ndarray) -> np.ndarray:
        bpoly = self.basis.evaluate(x_at_origin_subset, np.zeros((1, self.s_dim), dtype=int), True).T
        top = self.phs_rbf(r_rhs, sp.spline_degree).T
        out = np.empty((top.shape[0] + bpoly.shape[0], top.shape[1]), dtype=float)
        out[: top.shape[0], :] = top
        out[top.shape[0] :, :] = bpoly
        return out

    def _compute_weights_interior(self, x: np.ndarray, sp: StencilProperties, op: OpProperties, apply_op: str | Callable[..., np.ndarray], rhs_indices: int | np.ndarray) -> np.ndarray:
        # Build the operator right-hand side for the requested stencil rows and
        # solve the augmented RBF system to recover the final differentiation
        # weights.
        self.initialize_geometry(x, sp)
        rhs_inds = np.atleast_1d(rhs_indices).astype(int) - 1
        x_subset = x[rhs_inds]
        x_at_origin_subset = self.xc[rhs_inds]
        r_rhs = self.r_stencil[rhs_inds]
        b = self._apply_operator(apply_op, sp, op, r_rhs, x_subset, x, x_at_origin_subset, self.xc)
        if op.nosolve:
            w = b
        else:
            w = self.stable_solve(self.solve_lhs, b)[: self.n]
        if rhs_inds.size == 1 and rhs_inds[0] == 0:
            name = str(apply_op).lower()
            if name in {"lap", "laplacian"}:
                self.l = w
            elif name in {"interp", "interpolation"}:
                self.wt = w
            elif name in {"grad", "gradient"}:
                self.gx = w
        return w

    def _compute_weights_boundary(self, x: np.ndarray, nr: np.ndarray, neu_coeff: float, dir_coeff: float, sp: StencilProperties, op: OpProperties, apply_op: str | Callable[..., np.ndarray], rhs_indices: int | np.ndarray) -> np.ndarray:
        self.initialize_geometry(x, sp)
        rhs_inds = np.atleast_1d(rhs_indices).astype(int) - 1
        x_subset = x[rhs_inds]
        x_at_origin_subset = self.xc[rhs_inds]
        nr_subset = nr[rhs_inds]
        r_rhs = self.r_stencil[rhs_inds]
        b = self._apply_boundary_operator(apply_op, sp, op, neu_coeff, dir_coeff, r_rhs, x_subset, x, x_at_origin_subset, self.xc, nr_subset)
        w = self.stable_solve(self.solve_lhs, b)[: self.n]
        if rhs_inds.size == 1 and rhs_inds[0] == 0:
            self.bc = w
        return w

    def _apply_operator(self, apply_op: str | Callable[..., np.ndarray], sp: StencilProperties, op: OpProperties, r_rhs: np.ndarray, x_subset: np.ndarray, x: np.ndarray, x_at_origin_subset: np.ndarray, x_at_origin: np.ndarray) -> np.ndarray:
        if callable(apply_op):
            return apply_op(self, sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        name = str(apply_op).lower()
        if name in {"lap", "laplacian"}:
            return self.lap_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        if name in {"grad", "gradient"}:
            return self.grad_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        if name in {"interp", "interpolation"}:
            return self.interp_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        raise ValueError(f"unknown operator {apply_op}")

    def _apply_boundary_operator(self, apply_op: str | Callable[..., np.ndarray], sp: StencilProperties, op: OpProperties, neu_coeff: float, dir_coeff: float, r_rhs: np.ndarray, x_subset: np.ndarray, x: np.ndarray, x_at_origin_subset: np.ndarray, x_at_origin: np.ndarray, nr_subset: np.ndarray) -> np.ndarray:
        if callable(apply_op):
            return apply_op(self, sp, op, neu_coeff, dir_coeff, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin, nr_subset)
        name = str(apply_op).lower()
        if name in {"bc", "boundary"}:
            return self.bc_op(sp, op, neu_coeff, dir_coeff, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin, nr_subset)
        raise ValueError(f"unknown boundary operator {apply_op}")

    @staticmethod
    def phs_rbf(r: np.ndarray, degree: int) -> np.ndarray:
        return phs_kernel_matrix(r, degree)

    @staticmethod
    def phs_dr_over_r(r: np.ndarray, degree: int) -> np.ndarray:
        return phs_dr_over_r_matrix(r, degree)

    @staticmethod
    def phs_lap(r: np.ndarray, degree: int, dim: int) -> np.ndarray:
        return phs_lap_matrix(r, degree, dim)

    @staticmethod
    def stable_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Normal stencils should solve cleanly with a dense direct solve. The
        # pseudo-inverse fallback keeps the port alive on nearly singular cases
        # that occasionally arise in tiny or pathological stencils.
        try:
            x = dense_linalg.solve(a, b, assume_a="sym", check_finite=False, overwrite_b=False)
        except dense_linalg.LinAlgError:
            x = np.linalg.solve(a, b)
        if np.any(~np.isfinite(x)):
            x = dense_linalg.pinv(a, check_finite=False) @ b
        x[~np.isfinite(x)] = 0.0
        return x


@dataclass
class WeightedLeastSquaresStencil:
    x_stencil: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xc: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    width: float = 1.0
    s_dim: int = 0
    n: int = 0
    fit_ell: int = 0
    fit_npoly: int = 0
    node_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    interp_metric: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    reconstructor: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    basis: PolynomialBasis | None = None
    wt: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    l: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    bc: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def initialize_geometry(self, x: np.ndarray, sp: StencilProperties) -> None:
        # The WLS backend fits the same polynomial space as the RBF backend but
        # replaces the augmented interpolation solve with a weighted local least
        # squares reconstruction matrix.
        self.s_dim = x.shape[1]
        self.n = x.shape[0]
        self.fit_ell = sp.ell
        self.fit_npoly = sp.npoly
        self.x_stencil = x
        self.xm = x[0]
        r2 = np.sum((x - self.xm) ** 2, axis=1)
        self.width = max(float(np.sqrt(r2.max(initial=0.0))), 1.0)
        self.xc = (x - self.xm) / self.width
        self.basis = _basis_template(self.s_dim, self.fit_ell, "legendre")
        p = self.basis.evaluate(self.xc, _zero_derivative(self.s_dim), True)
        self.node_weights = np.clip(np.exp(-4.0 * r2 / (self.width**2)), 1e-10, 1.0)
        sqrtw = np.sqrt(self.node_weights)
        weighted_p = p * sqrtw[:, None]
        weighted_identity = np.diag(sqrtw)
        self.reconstructor = np.linalg.lstsq(weighted_p, weighted_identity, rcond=None)[0]
        if np.any(~np.isfinite(self.reconstructor)):
            self.reconstructor = np.linalg.pinv(weighted_p) @ weighted_identity
        self.reconstructor[~np.isfinite(self.reconstructor)] = 0.0
        self.interp_metric = weighted_p.T @ weighted_p

    def compute_weights(self, x: np.ndarray, *args: object) -> np.ndarray:
        if len(args) >= 7 and isinstance(args[0], np.ndarray) and args[0].shape[1] == x.shape[1]:
            nr, neu_coeff, dir_coeff, sp, op, apply_op, rhs_indices = args[:7]
            return self._compute_weights_boundary(x, nr, float(neu_coeff), float(dir_coeff), sp, op, apply_op, rhs_indices)
        sp, op, apply_op, rhs_indices = args[:4]
        return self._compute_weights_interior(x, sp, op, apply_op, rhs_indices)

    def lap_op(self, _sp: StencilProperties, _op: OpProperties, _r_rhs: np.ndarray, _x_subset: np.ndarray, _x: np.ndarray, x_at_origin_subset: np.ndarray, _x_at_origin: np.ndarray) -> np.ndarray:
        total = np.zeros((x_at_origin_subset.shape[0], self.fit_npoly))
        for d in range(self.s_dim):
            dd = np.zeros((1, self.s_dim), dtype=int)
            dd[0, d] = 2
            total += self.basis.evaluate(x_at_origin_subset, dd, True) / (self.width**2)
        return total.T

    def grad_op(self, _sp: StencilProperties, op: OpProperties, _r_rhs: np.ndarray, _x_subset: np.ndarray, _x: np.ndarray, x_at_origin_subset: np.ndarray, _x_at_origin: np.ndarray) -> np.ndarray:
        return self.basis.evaluate(x_at_origin_subset, _unit_multi_index(self.s_dim, op.selectdim), True).T / self.width

    def bc_op(self, _sp: StencilProperties, _op: OpProperties, neu_coeff: float, dir_coeff: float, _r_rhs: np.ndarray, _x_subset: np.ndarray, _x: np.ndarray, x_at_origin_subset: np.ndarray, _x_at_origin: np.ndarray, nr_subset: np.ndarray) -> np.ndarray:
        total = np.zeros((self.fit_npoly, x_at_origin_subset.shape[0]))
        if neu_coeff != 0:
            for d in range(self.s_dim):
                grad = self.basis.evaluate(x_at_origin_subset, _unit_multi_index(self.s_dim, d), True)
                total += neu_coeff * (grad.T * nr_subset[:, d]) / self.width
        if dir_coeff != 0:
            total += dir_coeff * self.basis.evaluate(x_at_origin_subset, np.zeros((1, self.s_dim), dtype=int), True).T
        if neu_coeff == 0 and dir_coeff == 0:
            raise ValueError("both boundary coefficients cannot be zero")
        return total

    def interp_op(self, _sp: StencilProperties, _op: OpProperties, _r_rhs: np.ndarray, _x_subset: np.ndarray, _x: np.ndarray, x_at_origin_subset: np.ndarray, _x_at_origin: np.ndarray) -> np.ndarray:
        return self.basis.evaluate(x_at_origin_subset, np.zeros((1, self.s_dim), dtype=int), True).T

    def _compute_weights_interior(self, x: np.ndarray, sp: StencilProperties, op: OpProperties, apply_op: str | Callable[..., np.ndarray], rhs_indices: int | np.ndarray) -> np.ndarray:
        # Apply the precomputed reconstructor to the polynomial target rows for
        # the requested operator.
        self.initialize_geometry(x, sp)
        rhs_inds = np.atleast_1d(rhs_indices).astype(int) - 1
        x_subset = x[rhs_inds]
        x_at_origin_subset = self.xc[rhs_inds]
        bpoly = self._apply_operator(apply_op, sp, op, None, x_subset, x, x_at_origin_subset, self.xc)
        w = self.reconstructor.T @ bpoly
        if rhs_inds.size == 1 and rhs_inds[0] == 0:
            name = str(apply_op).lower()
            if name in {"lap", "laplacian"}:
                self.l = w
            elif name in {"interp", "interpolation"}:
                self.wt = w
        return w

    def _compute_weights_boundary(self, x: np.ndarray, nr: np.ndarray, neu_coeff: float, dir_coeff: float, sp: StencilProperties, op: OpProperties, apply_op: str | Callable[..., np.ndarray], rhs_indices: int | np.ndarray) -> np.ndarray:
        self.initialize_geometry(x, sp)
        rhs_inds = np.atleast_1d(rhs_indices).astype(int) - 1
        x_subset = x[rhs_inds]
        x_at_origin_subset = self.xc[rhs_inds]
        nr_subset = nr[rhs_inds]
        bpoly = self._apply_boundary_operator(apply_op, sp, op, neu_coeff, dir_coeff, None, x_subset, x, x_at_origin_subset, self.xc, nr_subset)
        w = self.reconstructor.T @ bpoly
        if rhs_inds.size == 1 and rhs_inds[0] == 0:
            self.bc = w
        return w

    def _apply_operator(self, apply_op: str | Callable[..., np.ndarray], sp: StencilProperties, op: OpProperties, r_rhs: np.ndarray | None, x_subset: np.ndarray, x: np.ndarray, x_at_origin_subset: np.ndarray, x_at_origin: np.ndarray) -> np.ndarray:
        if callable(apply_op):
            return apply_op(self, sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        name = str(apply_op).lower()
        if name in {"lap", "laplacian"}:
            return self.lap_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        if name in {"grad", "gradient"}:
            return self.grad_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        if name in {"interp", "interpolation"}:
            return self.interp_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        raise ValueError(f"unknown operator {apply_op}")

    def _apply_boundary_operator(self, apply_op: str | Callable[..., np.ndarray], sp: StencilProperties, op: OpProperties, neu_coeff: float, dir_coeff: float, r_rhs: np.ndarray | None, x_subset: np.ndarray, x: np.ndarray, x_at_origin_subset: np.ndarray, x_at_origin: np.ndarray, nr_subset: np.ndarray) -> np.ndarray:
        if callable(apply_op):
            return apply_op(self, sp, op, neu_coeff, dir_coeff, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin, nr_subset)
        name = str(apply_op).lower()
        if name in {"bc", "boundary"}:
            return self.bc_op(sp, op, neu_coeff, dir_coeff, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin, nr_subset)
        raise ValueError(f"unknown boundary operator {apply_op}")


@dataclass
class FDDiffOp:
    approx_factory: Callable[[], object] = RBFStencil
    locations: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n1: int = 0
    n2: int = 0
    stencils: list[dict[str, object]] = field(default_factory=list)
    recorded_stencil_centers: list[np.ndarray] = field(default_factory=list)
    recorded_stencil_globals: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def assemble_op(self, domain: DomainDescriptor, op_name: str, st_props: StencilProperties, op_props: OpProperties, *, neu_coeff: np.ndarray | None = None, dir_coeff: np.ndarray | None = None, active_rows: np.ndarray | None = None) -> None:
        # Assemble one sparse operator row per requested center. Each row is
        # built from a local stencil query, then scattered into global triplet
        # form before converting to CSR.
        center_points, center_row_ids, center_col_globals, center_normals = _pick_centers(domain, st_props.point_set)
        if active_rows is None:
            active_rows = np.arange(1, center_points.shape[0] + 1)
        active_rows = np.asarray(active_rows, dtype=int)
        stencil_points = domain.get_tree_points(st_props.tree_mode)
        stencil_globals = domain.get_tree_globals(st_props.tree_mode)
        knn_indices = _query_center_stencils(domain, st_props.tree_mode, center_points, st_props.n)
        self.n1 = int(center_row_ids.max(initial=0))
        self.n2 = int(stencil_globals.max(initial=0))
        use_boundary = neu_coeff is not None or dir_coeff is not None
        all_indices = []
        all_weights = []
        all_stencils = []
        centers_recorded = []
        globals_recorded = np.zeros(active_rows.size, dtype=int)
        row_ids = np.zeros(active_rows.size, dtype=int)
        for k, local_row in enumerate(active_rows):
            indices, w, stencil, center_point, row_id, global_id = _assemble_one(
                self.approx_factory,
                stencil_points,
                knn_indices[int(local_row) - 1],
                center_points,
                center_row_ids,
                center_col_globals,
                center_normals,
                int(local_row),
                st_props,
                op_props,
                op_name,
                use_boundary,
                neu_coeff,
                dir_coeff,
            )
            all_indices.append(indices)
            all_weights.append(w)
            all_stencils.append(stencil)
            centers_recorded.append(center_point)
            globals_recorded[k] = global_id
            row_ids[k] = row_id
        triplet_count = active_rows.size * st_props.n
        self.locations = np.zeros((triplet_count, 2), dtype=int)
        self.values = np.zeros(triplet_count)
        cursor = 0
        self.stencils = []
        self.recorded_stencil_centers = []
        self.recorded_stencil_globals = globals_recorded
        for k in range(active_rows.size):
            cols = all_indices[k]
            w = all_weights[k]
            row_global = row_ids[k]
            next_cursor = cursor + cols.size
            self.locations[cursor:next_cursor, 0] = row_global
            self.locations[cursor:next_cursor, 1] = stencil_globals[cols - 1]
            self.values[cursor:next_cursor] = w[: cols.size, 0]
            cursor = next_cursor
            self.recorded_stencil_centers.append(centers_recorded[k])
            if op_props.record_stencils:
                self.stencils.append({"Approx": all_stencils[k], "Indices": stencil_globals[cols - 1]})
        self.locations = self.locations[:cursor]
        self.values = self.values[:cursor]

    def get_op(self) -> sparse.csr_matrix:
        if self.locations.size == 0:
            return sparse.csr_matrix((self.n1, self.n2))
        rows = self.locations[:, 0] - 1
        cols = self.locations[:, 1] - 1
        return sparse.csr_matrix((self.values, (rows, cols)), shape=(self.n1, self.n2))


@dataclass
class FDODiffOp(FDDiffOp):
    def assemble_op(self, domain: DomainDescriptor, op_name: str, st_props: StencilProperties, op_props: OpProperties, *, neu_coeff: np.ndarray | None = None, dir_coeff: np.ndarray | None = None, active_rows: np.ndarray | None = None) -> None:
        # The overlapped assembler reuses one stencil to write several nearby
        # rows, accepting only rows whose stability metrics do not exceed the
        # center row's metrics.
        center_points, center_row_ids, center_col_globals, center_normals = _pick_centers(domain, st_props.point_set)
        stencil_points = domain.get_tree_points(st_props.tree_mode)
        stencil_globals = domain.get_tree_globals(st_props.tree_mode)
        if active_rows is None:
            active_rows = np.arange(1, center_points.shape[0] + 1)
        active_rows = np.asarray(active_rows, dtype=int)
        active_set = np.zeros(center_points.shape[0], dtype=bool)
        active_set[active_rows - 1] = True
        active_remaining = int(np.count_nonzero(active_set))
        use_boundary = neu_coeff is not None or dir_coeff is not None
        loc_lim = max(1, int(np.floor(op_props.overlap_load * st_props.n)))
        row_to_local = _build_global_to_local_map(center_col_globals, int(stencil_globals.max(initial=0)))
        knn_indices = _query_center_stencils(domain, st_props.tree_mode, center_points, st_props.n)
        self.n1 = int(center_row_ids.max(initial=0))
        self.n2 = int(stencil_globals.max(initial=0))
        triplet_locations = np.zeros((active_rows.size * st_props.n, 2), dtype=int)
        triplet_values = np.zeros(active_rows.size * st_props.n)
        cursor = 0
        self.stencils = []
        self.recorded_stencil_centers = []
        recorded_globals: list[int] = []
        next_active = int(active_rows.min(initial=1) - 1)
        while active_remaining > 0:
            while next_active < active_set.size and not active_set[next_active]:
                next_active += 1
            if next_active >= active_set.size:
                raise RuntimeError("active row bookkeeping became inconsistent")
            local_center = next_active
            center_point = center_points[local_center]
            indices = _promote_center_to_front(knn_indices[local_center], int(center_col_globals[local_center]), stencil_globals)
            local_limit = min(loc_lim, indices.size)
            candidate_globals = stencil_globals[indices[:local_limit] - 1]
            candidate_locals = row_to_local[candidate_globals]
            candidate_active = candidate_locals >= 0
            if np.any(candidate_active):
                candidate_active[candidate_active] &= active_set[candidate_locals[candidate_active]]
            candidate_active[0] = True
            rhs_positions = np.flatnonzero(candidate_active)
            rhs_indices = rhs_positions + 1
            loc_x = stencil_points[indices - 1]
            stencil = self.approx_factory()
            if use_boundary:
                w = stencil.compute_weights(loc_x, center_normals[local_center : local_center + 1], float(neu_coeff[local_center]), float(dir_coeff[local_center]), st_props, op_props, op_name, rhs_indices)
            else:
                w = stencil.compute_weights(loc_x, st_props, op_props, op_name, rhs_indices)
            a = stencil.get_interp_mat()
            lebesgue = np.sum(np.abs(w[: st_props.n]), axis=0)
            aw = a @ w
            native = np.abs(np.sum(w * aw, axis=0))
            leb0 = lebesgue[0]
            nat0 = native[0]
            accepted_any = False
            for col_idx, j in enumerate(rhs_positions):
                candidate_local = int(candidate_locals[j])
                if candidate_local < 0 or not active_set[candidate_local]:
                    continue
                if lebesgue[col_idx] > leb0 or native[col_idx] > nat0:
                    continue
                next_cursor = cursor + indices.size
                triplet_locations[cursor:next_cursor, 0] = center_row_ids[candidate_local]
                triplet_locations[cursor:next_cursor, 1] = stencil_globals[indices - 1]
                triplet_values[cursor:next_cursor] = w[: indices.size, col_idx]
                cursor = next_cursor
                active_set[candidate_local] = False
                active_remaining -= 1
                if candidate_local < next_active:
                    next_active = candidate_local
                accepted_any = True
            if not accepted_any:
                raise RuntimeError("current overlapped stencil could not accept any active row")
            self.recorded_stencil_centers.append(center_point)
            recorded_globals.append(int(center_col_globals[local_center]))
            if op_props.record_stencils:
                self.stencils.append({"Approx": stencil, "Indices": stencil_globals[indices - 1]})
        self.locations = triplet_locations[:cursor]
        self.values = triplet_values[:cursor]
        self.recorded_stencil_globals = np.asarray(recorded_globals, dtype=int)


def _assemble_one(
    approx_factory: Callable[[], object],
    stencil_points: np.ndarray,
    indices: np.ndarray,
    center_points: np.ndarray,
    center_row_ids: np.ndarray,
    center_col_globals: np.ndarray,
    center_normals: np.ndarray | None,
    local_row: int,
    st_props: StencilProperties,
    op_props: OpProperties,
    op_name: str,
    use_boundary: bool,
    neu_coeff: np.ndarray | None,
    dir_coeff: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, object, np.ndarray, int, int]:
    # Gather the local stencil for one row and delegate the actual weight
    # construction to the selected approximation backend.
    center_point = center_points[local_row - 1]
    center_row_id = int(center_row_ids[local_row - 1])
    center_col_global = int(center_col_globals[local_row - 1])
    loc_x = stencil_points[indices - 1]
    stencil = approx_factory()
    if use_boundary:
        loc_nr = center_normals[local_row - 1 : local_row]
        w = stencil.compute_weights(loc_x, loc_nr, float(neu_coeff[local_row - 1]), float(dir_coeff[local_row - 1]), st_props, op_props, op_name, 1)
    else:
        w = stencil.compute_weights(loc_x, st_props, op_props, op_name, 1)
    return indices, w, stencil, center_point, center_row_id, center_col_global


def _query_center_stencils(domain: DomainDescriptor, tree_mode: str, center_points: np.ndarray, stencil_size: int) -> np.ndarray:
    indices, _ = domain.query_knn(tree_mode, center_points, stencil_size)
    return np.asarray(indices, dtype=int) + 1


def _build_global_to_local_map(center_col_globals: np.ndarray, max_global: int) -> np.ndarray:
    out = np.full(max(max_global, int(center_col_globals.max(initial=0))) + 1, -1, dtype=int)
    out[center_col_globals] = np.arange(center_col_globals.size, dtype=int)
    return out


def _promote_center_to_front(indices: np.ndarray, center_col_global: int, stencil_globals: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return indices
    center_matches = np.flatnonzero(stencil_globals[indices - 1] == center_col_global)
    if center_matches.size == 0:
        raise RuntimeError("overlapped stencil center is not present in its own neighbor list")
    center_pos = int(center_matches[0])
    if center_pos == 0:
        return indices
    out = indices.copy()
    out[0], out[center_pos] = out[center_pos], out[0]
    return out


def _pick_centers(domain: DomainDescriptor, point_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    # Map the user-facing point-set choice onto the actual node arrays and
    # global numbering used by the sparse operator assembly.
    normals = None
    mode = StencilProperties.normalize_point_set(point_set)
    if mode == "all":
        points = domain.get_all_nodes()
        row_ids = np.arange(1, points.shape[0] + 1)
        col_globals = row_ids.copy()
    elif mode == "interior_boundary":
        points = domain.get_int_bdry_nodes()
        row_ids = np.arange(1, points.shape[0] + 1)
        col_globals = row_ids.copy()
    elif mode == "boundary":
        points = domain.get_bdry_nodes()
        ni = domain.get_num_interior_nodes()
        row_ids = np.arange(1, points.shape[0] + 1)
        col_globals = ni + np.arange(1, points.shape[0] + 1)
        normals = domain.get_nrmls()
    else:
        raise ValueError("unknown point set")
    return points, row_ids, col_globals, normals
