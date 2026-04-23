from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Callable

import numpy as np
from scipy import sparse

from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import distance_matrix
from kernelpack.poly import PolynomialBasis, total_degree_indices


def _unit_multi_index(dim: int, selectdim: int) -> np.ndarray:
    d = np.zeros((1, dim), dtype=int)
    d[0, selectdim] = 1
    return d


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
                spline_degree += 1
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
    overlap_load: float = 0.5
    use_parallel: bool = False


@dataclass
class RBFStencil:
    a: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    solve_lhs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    x_stencil: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
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
        self.coeffs_already_computed = False
        self.s_dim = x.shape[1]
        self.n = x.shape[0]
        self.x_stencil = x
        self.ell = sp.ell
        self.npoly = sp.npoly
        r = distance_matrix(x, x)
        self.width = max(float(r.max(initial=0.0)), 1.0)
        self.xm = x.mean(axis=0)
        self.xc = (x - self.xm) / self.width
        self.basis = PolynomialBasis.from_total_degree(self.s_dim, self.ell, family="legendre", center=np.zeros(self.s_dim), scale=1.0)
        p = self.basis.evaluate(self.xc, np.zeros((1, self.s_dim), dtype=int), True)
        self.a = np.zeros((self.n + self.npoly, self.n + self.npoly))
        self.a[: self.n, : self.n] = self.phs_rbf(r, sp.spline_degree)
        self.a[: self.n, self.n :] = p
        self.a[self.n :, : self.n] = p.T
        self.solve_lhs = self.a

    def compute_weights(self, x: np.ndarray, *args: object) -> np.ndarray:
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
            dd = np.zeros((1, self.s_dim), dtype=int)
            dd[0, d] = 2
            bpoly += self.basis.evaluate(x_at_origin_subset, dd, True).T / (self.width**2)
        return np.vstack([self.phs_lap(r_rhs, sp.spline_degree, self.s_dim).T, bpoly])

    def grad_op(self, sp: StencilProperties, op: OpProperties, r_rhs: np.ndarray, x_subset: np.ndarray, x: np.ndarray, x_at_origin_subset: np.ndarray, _x_at_origin: np.ndarray) -> np.ndarray:
        dim = op.selectdim
        diff = x_subset[:, dim : dim + 1] - x[None, :, dim]
        bpoly = self.basis.evaluate(x_at_origin_subset, _unit_multi_index(self.s_dim, dim), True).T / self.width
        return np.vstack([(diff * self.phs_dr_over_r(r_rhs, sp.spline_degree)).T, bpoly])

    def bc_op(self, sp: StencilProperties, op: OpProperties, neu_coeff: float, dir_coeff: float, r_rhs: np.ndarray, x_subset: np.ndarray, x: np.ndarray, x_at_origin_subset: np.ndarray, x_at_origin: np.ndarray, nr_subset: np.ndarray) -> np.ndarray:
        total = np.zeros((self.n + self.npoly, x_at_origin_subset.shape[0]))
        if neu_coeff != 0:
            for d in range(self.s_dim):
                diff = x_subset[:, d : d + 1] - x[None, :, d]
                grad_rbf = (diff * self.phs_dr_over_r(r_rhs, sp.spline_degree)).T
                grad_poly = self.basis.evaluate(x_at_origin_subset, _unit_multi_index(self.s_dim, d), True).T / self.width
                total += neu_coeff * np.vstack([grad_rbf, grad_poly]) * nr_subset[:, d]
        if dir_coeff != 0:
            total += dir_coeff * self.interp_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        if dir_coeff == 0 and neu_coeff == 0:
            raise ValueError("both boundary coefficients cannot be zero")
        return total

    def interp_op(self, sp: StencilProperties, _op: OpProperties, r_rhs: np.ndarray, _x_subset: np.ndarray, _x: np.ndarray, x_at_origin_subset: np.ndarray, _x_at_origin: np.ndarray) -> np.ndarray:
        bpoly = self.basis.evaluate(x_at_origin_subset, np.zeros((1, self.s_dim), dtype=int), True).T
        return np.vstack([self.phs_rbf(r_rhs, sp.spline_degree).T, bpoly])

    def _compute_weights_interior(self, x: np.ndarray, sp: StencilProperties, op: OpProperties, apply_op: str | Callable[..., np.ndarray], rhs_indices: int | np.ndarray) -> np.ndarray:
        self.initialize_geometry(x, sp)
        rhs_inds = np.atleast_1d(rhs_indices).astype(int) - 1
        x_subset = x[rhs_inds]
        x_at_origin_subset = self.xc[rhs_inds]
        r = distance_matrix(x, x)
        r_rhs = r[rhs_inds]
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
        r = distance_matrix(x, x)
        r_rhs = r[rhs_inds]
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
        return np.where((degree % 2 == 0) & (r > 0), r**degree * np.log(r + 2e-16), r**degree)

    @staticmethod
    def phs_dr_over_r(r: np.ndarray, degree: int) -> np.ndarray:
        if degree % 2 == 0:
            d = r ** (degree - 2) * (degree * np.log(r + 2e-16) + 1)
        else:
            d = degree * r ** (degree - 2)
        d[~np.isfinite(d)] = 0.0
        return d

    @staticmethod
    def phs_lap(r: np.ndarray, degree: int, dim: int) -> np.ndarray:
        if degree % 2 == 0:
            logt = np.log(r + 2e-16)
            l = r ** (degree - 2) * (dim + 2 * degree + degree**2 * logt - 2 * degree * logt + dim * degree * logt - 2)
        else:
            l = degree * (dim + degree - 2) * r ** (degree - 2)
        l[~np.isfinite(l)] = 0.0
        return l

    @staticmethod
    def stable_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        x = np.linalg.solve(a, b)
        if np.any(~np.isfinite(x)):
            x = np.linalg.pinv(a) @ b
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
        self.s_dim = x.shape[1]
        self.n = x.shape[0]
        self.fit_ell = sp.ell
        self.fit_npoly = sp.npoly
        self.x_stencil = x
        self.xm = x[0]
        r2 = np.sum((x - self.xm) ** 2, axis=1)
        self.width = max(float(np.sqrt(r2.max(initial=0.0))), 1.0)
        self.xc = (x - self.xm) / self.width
        self.basis = PolynomialBasis.from_total_degree(self.s_dim, self.fit_ell, family="legendre", center=np.zeros(self.s_dim), scale=1.0)
        p = self.basis.evaluate(self.xc, np.zeros((1, self.s_dim), dtype=int), True)
        self.node_weights = np.clip(np.exp(-4.0 * r2 / (self.width**2)), 1e-10, 1.0)
        sqrtw = np.sqrt(self.node_weights)
        weighted_p = p * sqrtw[:, None]
        weighted_identity = np.diag(sqrtw)
        gram = weighted_p.T @ weighted_p
        if np.linalg.matrix_rank(weighted_p) < self.fit_npoly or np.linalg.cond(gram) > 1e12:
            self.reconstructor = np.linalg.pinv(weighted_p) @ weighted_identity
        else:
            self.reconstructor = np.linalg.lstsq(weighted_p, weighted_identity, rcond=None)[0]
        self.reconstructor[~np.isfinite(self.reconstructor)] = 0.0
        self.interp_metric = gram

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
        center_points, center_row_ids, center_col_globals, center_normals = _pick_centers(domain, st_props.point_set)
        if active_rows is None:
            active_rows = np.arange(1, center_points.shape[0] + 1)
        active_rows = np.asarray(active_rows, dtype=int)
        stencil_globals = domain.get_tree_globals(st_props.tree_mode)
        self.n1 = int(center_row_ids.max(initial=0))
        self.n2 = int(stencil_globals.max(initial=0))
        all_indices = []
        all_weights = []
        all_stencils = []
        centers_recorded = []
        globals_recorded = np.zeros(active_rows.size, dtype=int)
        row_ids = np.zeros(active_rows.size, dtype=int)
        use_boundary = neu_coeff is not None or dir_coeff is not None
        for k, local_row in enumerate(active_rows):
            indices, w, stencil, center_point, row_id, global_id = _assemble_one(
                self.approx_factory,
                domain,
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
            for j in range(st_props.n):
                self.locations[cursor] = [row_global, stencil_globals[cols[j] - 1]]
                self.values[cursor] = w[j, 0]
                cursor += 1
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
        center_points, center_row_ids, center_col_globals, center_normals = _pick_centers(domain, st_props.point_set)
        stencil_points = domain.get_tree_points(st_props.tree_mode)
        stencil_globals = domain.get_tree_globals(st_props.tree_mode)
        if active_rows is None:
            active_rows = np.arange(1, center_points.shape[0] + 1)
        active_rows = np.asarray(active_rows, dtype=int)
        active_set = np.zeros(center_points.shape[0], dtype=bool)
        active_set[active_rows - 1] = True
        use_boundary = neu_coeff is not None or dir_coeff is not None
        loc_lim = max(1, int(np.floor(op_props.overlap_load * st_props.n)))
        row_to_local = {int(center_col_globals[k]): k for k in range(center_points.shape[0])}
        self.n1 = int(center_row_ids.max(initial=0))
        self.n2 = int(stencil_globals.max(initial=0))
        triplet_locations = np.zeros((active_rows.size * st_props.n, 2), dtype=int)
        triplet_values = np.zeros(active_rows.size * st_props.n)
        cursor = 0
        self.stencils = []
        self.recorded_stencil_centers = []
        self.recorded_stencil_globals = np.zeros(0, dtype=int)
        while np.any(active_set):
            local_center = int(np.flatnonzero(active_set)[0])
            center_point = center_points[local_center]
            indices, _ = domain.query_knn(st_props.tree_mode, center_point, st_props.n)
            indices = indices[0] + 1
            rhs_indices = np.arange(1, min(loc_lim, indices.size) + 1)
            loc_x = stencil_points[indices - 1]
            stencil = self.approx_factory()
            if use_boundary:
                w = stencil.compute_weights(loc_x, center_normals[local_center : local_center + 1], float(neu_coeff[local_center]), float(dir_coeff[local_center]), st_props, op_props, op_name, rhs_indices)
            else:
                w = stencil.compute_weights(loc_x, st_props, op_props, op_name, rhs_indices)
            a = stencil.get_interp_mat()
            lebesgue = np.sum(np.abs(w[: st_props.n]), axis=0)
            native = np.array([abs(w[:, j].T @ (a @ w[:, j])) for j in range(w.shape[1])])
            leb0 = lebesgue[0]
            nat0 = native[0]
            accepted_any = False
            for j in range(min(loc_lim, indices.size)):
                candidate_col_global = stencil_globals[indices[j] - 1]
                candidate_local = row_to_local.get(int(candidate_col_global))
                if candidate_local is None or not active_set[candidate_local]:
                    continue
                if lebesgue[j] > leb0 or native[j] > nat0:
                    continue
                for q in range(st_props.n):
                    triplet_locations[cursor] = [center_row_ids[candidate_local], stencil_globals[indices[q] - 1]]
                    triplet_values[cursor] = w[q, j]
                    cursor += 1
                active_set[candidate_local] = False
                accepted_any = True
            if not accepted_any:
                raise RuntimeError("current overlapped stencil could not accept any active row")
            self.recorded_stencil_centers.append(center_point)
            self.recorded_stencil_globals = np.append(self.recorded_stencil_globals, center_col_globals[local_center])
            if op_props.record_stencils:
                self.stencils.append({"Approx": stencil, "Indices": stencil_globals[indices - 1]})
        self.locations = triplet_locations[:cursor]
        self.values = triplet_values[:cursor]


def _assemble_one(approx_factory: Callable[[], object], domain: DomainDescriptor, center_points: np.ndarray, center_row_ids: np.ndarray, center_col_globals: np.ndarray, center_normals: np.ndarray | None, local_row: int, st_props: StencilProperties, op_props: OpProperties, op_name: str, use_boundary: bool, neu_coeff: np.ndarray | None, dir_coeff: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, object, np.ndarray, int, int]:
    center_point = center_points[local_row - 1]
    center_row_id = int(center_row_ids[local_row - 1])
    center_col_global = int(center_col_globals[local_row - 1])
    indices, _ = domain.query_knn(st_props.tree_mode, center_point, st_props.n)
    indices = indices[0] + 1
    stencil_points = domain.get_tree_points(st_props.tree_mode)
    loc_x = stencil_points[indices - 1]
    stencil = approx_factory()
    if use_boundary:
        loc_nr = center_normals[local_row - 1 : local_row]
        w = stencil.compute_weights(loc_x, loc_nr, float(neu_coeff[local_row - 1]), float(dir_coeff[local_row - 1]), st_props, op_props, op_name, 1)
    else:
        w = stencil.compute_weights(loc_x, st_props, op_props, op_name, 1)
    return indices, w, stencil, center_point, center_row_id, center_col_global


def _pick_centers(domain: DomainDescriptor, point_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
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
