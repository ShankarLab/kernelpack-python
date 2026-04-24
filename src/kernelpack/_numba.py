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
