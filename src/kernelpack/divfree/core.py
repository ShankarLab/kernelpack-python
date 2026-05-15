from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import linalg as dense_linalg
from scipy.spatial import cKDTree

from kernelpack._numba import divfree_gram_matrix
from kernelpack import poly


def DFPHS(dim: int, m_in: int) -> tuple[tuple[tuple[Callable[..., np.ndarray], ...], ...], ...]:
    # Preserve the Matlab-facing prototype for callers that want callable block
    # entries, but route the heavy matrix work through direct block builders.
    dim = int(dim)
    m = int(m_in)
    if dim not in {2, 3}:
        raise ValueError("divergence-free PHS is only supported in 2D and 3D")
    if m <= 0 or (m % 2) == 0:
        raise ValueError("divergence-free PHS currently expects a positive odd degree")

    def _make_entry(i: int, j: int, mode: str) -> Callable[..., np.ndarray]:
        def _entry(r: np.ndarray, *coords: np.ndarray) -> np.ndarray:
            diff = _diff_tensor_from_coords(dim, r, i, j, coords)
            return _evaluate_dfphs_entry(dim, m, i, j, mode, r, diff)

        _entry._divfree_dim = dim  # type: ignore[attr-defined]
        _entry._divfree_degree = m  # type: ignore[attr-defined]
        return _entry

    dfrbf = tuple(tuple(_make_entry(i, j, "divfree") for j in range(dim)) for i in range(dim))
    lap_dfrbf = tuple(tuple(_make_entry(i, j, "lap_divfree") for j in range(dim)) for i in range(dim))
    hess_rbf = tuple(tuple(_make_entry(i, j, "hessian") for j in range(dim)) for i in range(dim))
    full_rbf = tuple(tuple(_make_entry(i, j, "full") for j in range(dim)) for i in range(dim))
    return dfrbf, lap_dfrbf, hess_rbf, full_rbf


def _diff_tensor_from_coords(
    dim: int,
    r: np.ndarray,
    i: int,
    j: int,
    coords: tuple[np.ndarray, ...],
) -> np.ndarray:
    diff = np.zeros(r.shape + (dim,), dtype=float)
    if i == j:
        if len(coords) != 2:
            raise ValueError("diagonal DFPHS entries expect two coordinate arrays")
        diff[:, :, i] = np.asarray(coords[0], dtype=float) - np.asarray(coords[1], dtype=float)
        return diff

    if len(coords) != 4:
        raise ValueError("off-diagonal DFPHS entries expect four coordinate arrays")
    diff[:, :, i] = np.asarray(coords[0], dtype=float) - np.asarray(coords[1], dtype=float)
    diff[:, :, j] = np.asarray(coords[2], dtype=float) - np.asarray(coords[3], dtype=float)
    return diff


def _evaluate_dfphs_entry(
    dim: int,
    m: int,
    i: int,
    j: int,
    mode: str,
    r: np.ndarray,
    diff: np.ndarray,
) -> np.ndarray:
    re = np.asarray(r, dtype=float) + np.finfo(float).eps
    lap_scalar = m * (m + dim - 2) * re ** (m - 2)
    full_kernel = np.zeros_like(re)
    if i == j:
        full_kernel = -lap_scalar

    hess_term = m * (m - 2) * re ** (m - 4) * diff[:, :, i] * diff[:, :, j]
    if i == j:
        hess_term = hess_term + m * re ** (m - 2)

    divfree_kernel = full_kernel + hess_term
    lap_divfree_kernel = np.zeros_like(re)
    if i == j:
        lap_divfree_kernel = m * (m - 2) * re ** (m - 4) * (2 - (m + dim - 3) * (m + dim - 4))
    lap_divfree_kernel = lap_divfree_kernel + m * (m - 2) * (m - 4) * (m + dim - 2) * re ** (m - 6) * diff[:, :, i] * diff[:, :, j]

    if mode == "divfree":
        return divfree_kernel
    if mode == "lap_divfree":
        return lap_divfree_kernel
    if mode == "hessian":
        return hess_term
    if mode == "full":
        return full_kernel
    raise ValueError("unknown DFPHS kernel mode")


def DivFreeGram(_dfrbf: object, X: np.ndarray, Y: np.ndarray, _r: np.ndarray | None = None, *, degree: int | None = None) -> np.ndarray:
    if degree is None:
        try:
            degree = int(_dfrbf[0][0]._divfree_degree)  # type: ignore[index, attr-defined]
        except Exception as exc:
            raise ValueError("DivFreeGram needs either a DFPHS kernel tuple or an explicit degree keyword") from exc
    return divfree_gram_matrix(np.asarray(X, dtype=float), np.asarray(Y, dtype=float), int(degree))


def _stack_field(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    return np.concatenate([U[:, d] for d in range(U.shape[1])], axis=0)


def _unstack_field(rhs: np.ndarray, dim: int) -> np.ndarray:
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    n = rhs.size // dim
    return np.column_stack([rhs[d * n : (d + 1) * n] for d in range(dim)])


def _apply_vec_rotation(Uref: np.ndarray, R: np.ndarray) -> np.ndarray:
    dim = R.shape[0]
    nstack = Uref.shape[0]
    if nstack % dim != 0:
        raise ValueError("stacked vector field length must be divisible by the dimension")
    n = nstack // dim
    blocks = [Uref[d * n : (d + 1) * n, :] for d in range(dim)]
    out = np.zeros_like(Uref)
    for i in range(dim):
        block = np.zeros((n, Uref.shape[1]), dtype=float)
        for j in range(dim):
            block += R[i, j] * blocks[j]
        out[i * n : (i + 1) * n, :] = block
    return out


@dataclass
class DivFreePolynomialData:
    d: int
    p_out: int
    pA: int
    aA: np.ndarray
    center: np.ndarray
    scale: float
    R: np.ndarray
    piv: np.ndarray
    R11: np.ndarray
    rank_tol: float
    recurrence: Callable[[int], tuple[np.ndarray, np.ndarray]]

    def eval(self, xq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        if xq.ndim != 2 or xq.shape[1] != self.d:
            raise ValueError(f"xq must have shape (n, {self.d})")
        xiq = ((xq - self.center) @ self.R) / self.scale
        Vq = _divfree_polynomial_stack(xiq, self.aA, self.recurrence)
        if self.piv.size == 0:
            return np.zeros((self.d * xq.shape[0], 0), dtype=float)
        Qref_q = dense_linalg.solve_triangular(self.R11.T, Vq[:, self.piv].T, lower=True, check_finite=False).T
        return _apply_vec_rotation(Qref_q, self.R)


def _divfree_polynomial_stack(
    x: np.ndarray,
    alpha: np.ndarray,
    recurrence: Callable[[int], tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    m, dim = x.shape
    if dim == 2:
        dphidx = poly.mpoly_eval(x, alpha, recurrence, np.array([[1, 0]], dtype=int))
        dphidy = poly.mpoly_eval(x, alpha, recurrence, np.array([[0, 1]], dtype=int))
        V = np.zeros((2 * m, alpha.shape[0]), dtype=float)
        V[:m, :] = dphidy
        V[m:, :] = -dphidx
        return V

    dphidx = poly.mpoly_eval(x, alpha, recurrence, np.array([[1, 0, 0]], dtype=int))
    dphidy = poly.mpoly_eval(x, alpha, recurrence, np.array([[0, 1, 0]], dtype=int))
    dphidz = poly.mpoly_eval(x, alpha, recurrence, np.array([[0, 0, 1]], dtype=int))
    K = alpha.shape[0]
    V = np.zeros((3 * m, 3 * K), dtype=float)
    V[m : 2 * m, :K] = dphidz
    V[2 * m :, :K] = -dphidy
    V[:m, K : 2 * K] = -dphidz
    V[2 * m :, K : 2 * K] = dphidx
    V[:m, 2 * K :] = dphidy
    V[m : 2 * m, 2 * K :] = -dphidx
    return V


def df_poly_basis_from_jacobi(
    p: np.ndarray,
    p_out: int,
    recurrence: Callable[[int], tuple[np.ndarray, np.ndarray]] | None = None,
    opts: dict[str, object] | None = None,
) -> tuple[np.ndarray, DivFreePolynomialData]:
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] not in {2, 3}:
        raise ValueError("p must have shape (m, 2) or (m, 3)")
    if recurrence is None:
        recurrence = lambda N: poly.jacobi_recurrence(N, 0.0, 0.0)
    if opts is None:
        opts = {}

    m, dim = p.shape
    center = np.asarray(opts.get("center", p.mean(axis=0)), dtype=float).reshape(dim)
    X = p - center
    scale = float(opts.get("scale", np.linalg.norm(X, axis=1).max(initial=1.0)))
    if not (scale > 0):
        scale = 1.0
    use_pca_frame = bool(opts.get("use_pca_frame", False))
    rank_tol = float(opts.get("rank_tol", 1e-12))

    R = np.eye(dim, dtype=float)
    if use_pca_frame:
        C = (X.T @ X) / max(m, 1)
        U, _, _ = np.linalg.svd(C)
        R = U
        for j in range(dim):
            k = int(np.argmax(np.abs(R[:, j])))
            if R[k, j] < 0:
                R[:, j] = -R[:, j]
        if np.linalg.det(R) < 0:
            R[:, -1] = -R[:, -1]

    xi = (X @ R) / scale
    pA = int(p_out) + 1
    aA = poly.total_degree_indices(dim, pA)
    V = _divfree_polynomial_stack(xi, aA, recurrence)
    Qref, Rq, piv = dense_linalg.qr(V, mode="economic", pivoting=True, check_finite=False)
    dd = np.abs(np.diag(Rq))
    if dd.size == 0 or dd[0] == 0:
        rank = 0
    else:
        keep = np.flatnonzero(dd > rank_tol * dd[0])
        rank = int(keep[-1] + 1) if keep.size else 0
    Qref = Qref[:, :rank]
    R11 = Rq[:rank, :rank]
    piv = np.asarray(piv[:rank], dtype=int)
    P = _apply_vec_rotation(Qref, R)
    poly_data = DivFreePolynomialData(
        d=dim,
        p_out=int(p_out),
        pA=pA,
        aA=aA,
        center=center,
        scale=scale,
        R=R,
        piv=piv,
        R11=R11,
        rank_tol=rank_tol,
        recurrence=recurrence,
    )
    return P, poly_data


@dataclass
class DivFreePHSInterpolant:
    Dim: int = 0
    PhsDegree: int = 0
    PolyDegree: int = 0
    Nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Values: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    RBFCoeffs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    PolyCoeffs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    SaddleMatrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    PolyData: DivFreePolynomialData | None = None

    @staticmethod
    def fit(
        X: np.ndarray,
        U: np.ndarray,
        polyDegree: int,
        phsDegree: int,
        *,
        Recurrence: Callable[[int], tuple[np.ndarray, np.ndarray]] | None = None,
        PolyOptions: dict[str, object] | None = None,
    ) -> "DivFreePHSInterpolant":
        obj = DivFreePHSInterpolant()
        obj.initialize(X, U, polyDegree, phsDegree, Recurrence=Recurrence, PolyOptions=PolyOptions)
        return obj

    def initialize(
        self,
        X: np.ndarray,
        U: np.ndarray,
        polyDegree: int,
        phsDegree: int,
        *,
        Recurrence: Callable[[int], tuple[np.ndarray, np.ndarray]] | None = None,
        PolyOptions: dict[str, object] | None = None,
    ) -> None:
        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)
        if X.ndim != 2 or U.ndim != 2 or U.shape != X.shape:
            raise ValueError("X and U must both have shape (n, dim)")
        if X.shape[1] not in {2, 3}:
            raise ValueError("divergence-free interpolation is only supported in 2D and 3D")
        if phsDegree <= 0 or (int(phsDegree) % 2) == 0:
            raise ValueError("divergence-free PHS currently expects a positive odd degree")
        if Recurrence is None:
            Recurrence = lambda N: poly.jacobi_recurrence(N, 0.0, 0.0)
        if PolyOptions is None:
            PolyOptions = {}

        self.Dim = int(X.shape[1])
        self.PhsDegree = int(phsDegree)
        self.PolyDegree = int(polyDegree)
        self.Nodes = X
        self.Values = U

        A = divfree_gram_matrix(X, X, self.PhsDegree)
        P, polydata = df_poly_basis_from_jacobi(X, self.PolyDegree, Recurrence, PolyOptions)
        rhs = _stack_field(U)
        saddle = np.zeros((A.shape[0] + P.shape[1], A.shape[1] + P.shape[1]), dtype=float)
        saddle[: A.shape[0], : A.shape[1]] = A
        saddle[: A.shape[0], A.shape[1] :] = P
        saddle[A.shape[1] :, : A.shape[1]] = P.T
        coeffs = dense_linalg.solve(saddle, np.concatenate([rhs, np.zeros(P.shape[1], dtype=float)]), assume_a="sym", check_finite=False)

        n_rbf = A.shape[1]
        self.RBFCoeffs = coeffs[:n_rbf]
        self.PolyCoeffs = coeffs[n_rbf:]
        self.PolyData = polydata
        self.SaddleMatrix = saddle

    def evaluate(self, Xq: np.ndarray) -> np.ndarray:
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim != 2 or Xq.shape[1] != self.Dim:
            raise ValueError(f"Xq must have shape (n, {self.Dim})")
        Aeval = divfree_gram_matrix(Xq, self.Nodes, self.PhsDegree)
        Pq = self.PolyData.eval(Xq) if self.PolyData is not None else np.zeros((self.Dim * Xq.shape[0], 0), dtype=float)
        stacked = Aeval @ self.RBFCoeffs
        if self.PolyCoeffs.size:
            stacked = stacked + Pq @ self.PolyCoeffs
        return _unstack_field(stacked, self.Dim)


@dataclass
class LocalDivFreeInterpolator:
    Nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Values: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    PolyDegree: int = 0
    PhsDegree: int = 0
    StencilSize: int = 0
    Tree: cKDTree | None = None
    CenterModels: list[DivFreePHSInterpolant | None] = field(default_factory=list)
    StencilIndices: list[np.ndarray | None] = field(default_factory=list)
    ActiveCenters: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    Recurrence: Callable[[int], tuple[np.ndarray, np.ndarray]] = field(default_factory=lambda: (lambda N: poly.jacobi_recurrence(N, 0.0, 0.0)))
    PolyOptions: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def fit(
        X: np.ndarray,
        U: np.ndarray,
        polyDegree: int,
        phsDegree: int,
        stencilSize: int,
        *,
        Recurrence: Callable[[int], tuple[np.ndarray, np.ndarray]] | None = None,
        PolyOptions: dict[str, object] | None = None,
        ActiveCenters: np.ndarray | None = None,
    ) -> "LocalDivFreeInterpolator":
        obj = LocalDivFreeInterpolator()
        obj.initialize(
            X,
            U,
            polyDegree,
            phsDegree,
            stencilSize,
            Recurrence=Recurrence,
            PolyOptions=PolyOptions,
            ActiveCenters=ActiveCenters,
        )
        return obj

    def initialize(
        self,
        X: np.ndarray,
        U: np.ndarray,
        polyDegree: int,
        phsDegree: int,
        stencilSize: int,
        *,
        Recurrence: Callable[[int], tuple[np.ndarray, np.ndarray]] | None = None,
        PolyOptions: dict[str, object] | None = None,
        ActiveCenters: np.ndarray | None = None,
    ) -> None:
        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)
        if X.ndim != 2 or U.ndim != 2 or U.shape != X.shape:
            raise ValueError("X and U must both have shape (n, dim)")
        if Recurrence is None:
            Recurrence = lambda N: poly.jacobi_recurrence(N, 0.0, 0.0)
        if PolyOptions is None:
            PolyOptions = {}

        self.Nodes = X
        self.Values = U
        self.PolyDegree = int(polyDegree)
        self.PhsDegree = int(phsDegree)
        self.StencilSize = min(int(stencilSize), X.shape[0])
        self.Recurrence = Recurrence
        self.PolyOptions = dict(PolyOptions)
        self.Tree = cKDTree(X)
        self.CenterModels = [None] * X.shape[0]
        self.StencilIndices = [None] * X.shape[0]
        if ActiveCenters is None:
            self.ActiveCenters = np.arange(X.shape[0], dtype=int)
        else:
            self.ActiveCenters = np.unique(np.asarray(ActiveCenters, dtype=int).reshape(-1))
        for j in self.ActiveCenters:
            self._build_center_model(int(j))

    def _query_stencil_indices(self, point: np.ndarray) -> np.ndarray:
        idx = self.Tree.query(point, k=self.StencilSize)[1]
        idx = np.asarray(idx, dtype=int).reshape(-1)
        return idx

    def _build_center_model(self, center_idx: int) -> DivFreePHSInterpolant:
        idx = self._query_stencil_indices(self.Nodes[center_idx])
        self.StencilIndices[center_idx] = idx
        model = DivFreePHSInterpolant.fit(
            self.Nodes[idx],
            self.Values[idx],
            self.PolyDegree,
            self.PhsDegree,
            Recurrence=self.Recurrence,
            PolyOptions=self.PolyOptions,
        )
        self.CenterModels[center_idx] = model
        return model

    def assign_centers(self, Xq: np.ndarray) -> np.ndarray:
        Xq = np.asarray(Xq, dtype=float)
        idx = self.Tree.query(Xq, k=1)[1]
        return np.asarray(idx, dtype=int).reshape(-1)

    def evaluate(self, Xq: np.ndarray, *, CenterIndices: np.ndarray | None = None) -> np.ndarray:
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim != 2 or Xq.shape[1] != self.Nodes.shape[1]:
            raise ValueError(f"Xq must have shape (n, {self.Nodes.shape[1]})")
        center_idx = self.assign_centers(Xq) if CenterIndices is None else np.asarray(CenterIndices, dtype=int).reshape(-1)
        if center_idx.size != Xq.shape[0]:
            raise ValueError("CenterIndices must match the number of query points")
        Uq = np.zeros((Xq.shape[0], self.Values.shape[1]), dtype=float)
        for j in np.unique(center_idx):
            mask = center_idx == j
            model = self.CenterModels[int(j)]
            if model is None:
                model = self._build_center_model(int(j))
            Uq[mask, :] = model.evaluate(Xq[mask])
        return Uq
