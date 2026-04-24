from __future__ import annotations

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path only matters when numba is absent
    njit = None
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _distance_matrix_numba(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty((x.shape[0], y.shape[0]), dtype=np.float64)
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                accum = 0.0
                for d in range(x.shape[1]):
                    diff = x[i, d] - y[j, d]
                    accum += diff * diff
                out[i, j] = np.sqrt(max(accum, 0.0))
        return out


    @njit(cache=True, fastmath=True)
    def _phs_kernel_numba(r: np.ndarray, degree: int) -> np.ndarray:
        out = np.empty_like(r)
        even_degree = (degree % 2) == 0
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                rij = r[i, j]
                if even_degree:
                    out[i, j] = (rij**degree) * np.log(rij + 2.0e-16) if rij > 0.0 else 0.0
                else:
                    out[i, j] = rij**degree
        return out


    @njit(cache=True, fastmath=True)
    def _phs_dr_over_r_numba(r: np.ndarray, degree: int) -> np.ndarray:
        out = np.empty_like(r)
        even_degree = (degree % 2) == 0
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                rij = r[i, j]
                if even_degree:
                    value = (rij ** (degree - 2)) * (degree * np.log(rij + 2.0e-16) + 1.0)
                else:
                    value = degree * (rij ** (degree - 2))
                out[i, j] = value if np.isfinite(value) else 0.0
        return out


    @njit(cache=True, fastmath=True)
    def _phs_lap_numba(r: np.ndarray, degree: int, dim: int) -> np.ndarray:
        out = np.empty_like(r)
        even_degree = (degree % 2) == 0
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                rij = r[i, j]
                if even_degree:
                    logt = np.log(rij + 2.0e-16)
                    value = (rij ** (degree - 2)) * (
                        dim + 2.0 * degree + degree * degree * logt - 2.0 * degree * logt + dim * degree * logt - 2.0
                    )
                else:
                    value = degree * (dim + degree - 2.0) * (rij ** (degree - 2))
                out[i, j] = value if np.isfinite(value) else 0.0
        return out

else:
    _distance_matrix_numba = None
    _phs_kernel_numba = None
    _phs_dr_over_r_numba = None
    _phs_lap_numba = None


def dense_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.zeros((x.shape[0], y.shape[0]), dtype=float)
    if NUMBA_AVAILABLE:
        return _distance_matrix_numba(x, y)
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    y_sq = np.sum(y * y, axis=1, keepdims=True).T
    sq_dist = np.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)
    return np.sqrt(sq_dist)


def phs_kernel_matrix(r: np.ndarray, degree: int) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return np.zeros_like(r)
    if NUMBA_AVAILABLE:
        return _phs_kernel_numba(r, int(degree))
    if degree % 2 == 0:
        return np.where(r > 0, r**degree * np.log(r + 2e-16), 0.0)
    return r**degree


def phs_dr_over_r_matrix(r: np.ndarray, degree: int) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return np.zeros_like(r)
    if NUMBA_AVAILABLE:
        return _phs_dr_over_r_numba(r, int(degree))
    if degree % 2 == 0:
        d = r ** (degree - 2) * (degree * np.log(r + 2e-16) + 1)
    else:
        d = degree * r ** (degree - 2)
    d[~np.isfinite(d)] = 0.0
    return d


def phs_lap_matrix(r: np.ndarray, degree: int, dim: int) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return np.zeros_like(r)
    if NUMBA_AVAILABLE:
        return _phs_lap_numba(r, int(degree), int(dim))
    if degree % 2 == 0:
        logt = np.log(r + 2e-16)
        l = r ** (degree - 2) * (dim + 2 * degree + degree**2 * logt - 2 * degree * logt + dim * degree * logt - 2)
    else:
        l = degree * (dim + degree - 2) * r ** (degree - 2)
    l[~np.isfinite(l)] = 0.0
    return l


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _legendre_recurrence_numba(n: int) -> tuple[np.ndarray, np.ndarray]:
        a = np.zeros(n, dtype=np.float64)
        b = np.ones(n, dtype=np.float64)
        if n > 0:
            b[0] = 2.0
        if n > 1:
            b[1] = 1.0 / 3.0
        for q in range(2, n):
            qq = float(q)
            b[q] = (qq * qq) / ((2.0 * qq + 1.0) * (2.0 * qq - 1.0))
        return a, b


    @njit(cache=True, fastmath=True)
    def _legendre_eval_numba(x: np.ndarray, n: int, d: int) -> np.ndarray:
        a, b = _legendre_recurrence_numba(n + 1)
        out = np.zeros((x.size, n + 1), dtype=np.float64)
        out[:, 0] = 1.0 / np.sqrt(b[0])
        if n > 0:
            out[:, 1] = ((x - a[0]) * out[:, 0]) / np.sqrt(b[1])
        for q in range(1, n):
            out[:, q + 1] = ((x - a[q]) * out[:, q] - np.sqrt(b[q]) * out[:, q - 1]) / np.sqrt(b[q + 1])

        if d == 0:
            return out

        cur = out
        for qd in range(1, d + 1):
            nxt = np.zeros_like(cur)
            for q in range(qd, n + 1):
                if q == qd:
                    denom = 1.0
                    for j in range(q + 1):
                        denom *= b[j]
                    const = 1.0
                    for j in range(2, qd + 1):
                        const *= float(j)
                    nxt[:, q] = const / np.sqrt(denom)
                else:
                    nxt[:, q] = ((x - a[q - 1]) * nxt[:, q - 1] - np.sqrt(b[q - 1]) * nxt[:, q - 2] + qd * cur[:, q - 1]) / np.sqrt(b[q])
            cur = nxt
        return cur


    @njit(cache=True, fastmath=True)
    def _legendre_tensor_evaluate_numba(x: np.ndarray, alpha: np.ndarray, d: np.ndarray) -> np.ndarray:
        max_alpha = 0
        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                if alpha[i, j] > max_alpha:
                    max_alpha = alpha[i, j]
        a0 = np.sqrt(2.0)
        out = np.ones((x.shape[0], alpha.shape[0], d.shape[0]), dtype=np.float64) / (a0 ** x.shape[1])
        for qd in range(d.shape[0]):
            for qdim in range(x.shape[1]):
                local_max = 0
                for i in range(alpha.shape[0]):
                    if alpha[i, qdim] > local_max:
                        local_max = alpha[i, qdim]
                temp = _legendre_eval_numba(x[:, qdim], local_max, int(d[qd, qdim]))
                for i in range(alpha.shape[0]):
                    degree = alpha[i, qdim]
                    if d[qd, qdim] == 0 and degree == 0:
                        continue
                    out[:, i, qd] *= temp[:, degree] * a0
        return out

else:
    _legendre_tensor_evaluate_numba = None


def legendre_tensor_evaluate(x: np.ndarray, alpha: np.ndarray, d: np.ndarray) -> np.ndarray | None:
    if not NUMBA_AVAILABLE:
        return None
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    return _legendre_tensor_evaluate_numba(x, alpha, d)
