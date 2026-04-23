from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable

import numpy as np
from scipy.special import gammaln


class JacobiPolynomials:
    @staticmethod
    def recurrence(n: int, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
        n = int(np.max(np.atleast_1d(n)))
        idx = np.arange(n, dtype=float)
        a = (beta**2 - alpha**2) * np.ones_like(idx)
        b = np.ones_like(idx)

        flags0 = idx == 0
        if np.any(flags0):
            a[flags0] = (beta - alpha) / (alpha + beta + 2.0)
            b[flags0] = np.exp(
                (alpha + beta + 1.0) * np.log(2.0)
                + gammaln(alpha + 1.0)
                + gammaln(beta + 1.0)
                - gammaln(alpha + beta + 2.0)
            )

        flags1 = idx == 1
        if np.any(flags1):
            a[flags1] = a[flags1] / ((2.0 + alpha + beta) * (4.0 + alpha + beta))
            b[flags1] = (
                4.0
                * (1.0 + alpha)
                * (1.0 + beta)
                / ((2.0 + alpha + beta) ** 2 * (3.0 + alpha + beta))
            )

        flags = ~(flags0 | flags1)
        if np.any(flags):
            nloc = idx[flags]
            a[flags] = a[flags] / ((2.0 * nloc + alpha + beta) * (2.0 * nloc + alpha + beta + 2.0))
            b[flags] = (
                4.0
                * nloc
                * (nloc + alpha)
                * (nloc + beta)
                * (nloc + alpha + beta)
                / (
                    (2.0 * nloc + alpha + beta) ** 2
                    * (2.0 * nloc + alpha + beta + 1.0)
                    * (2.0 * nloc + alpha + beta - 1.0)
                )
            )
        return a, b

    @staticmethod
    def evaluate(a: np.ndarray, b: np.ndarray, x: np.ndarray, n: int, d: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        if d < 0 or n < 0:
            raise ValueError("n and d must be nonnegative")
        if n >= len(a) or n >= len(b):
            raise ValueError("recurrence length is too short")

        out = np.zeros((x.size, n + 1), dtype=float)
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
                    nxt[:, q] = np.exp(gammaln(qd + 1.0) - 0.5 * np.sum(np.log(b[: q + 1])))
                else:
                    nxt[:, q] = (
                        (x - a[q - 1]) * nxt[:, q - 1]
                        - np.sqrt(b[q - 1]) * nxt[:, q - 2]
                        + qd * cur[:, q - 1]
                    ) / np.sqrt(b[q])
            cur = nxt
        return cur

    @staticmethod
    def tensor_evaluate(
        x: np.ndarray,
        alpha: np.ndarray,
        recurrence_handle: Callable[[int], tuple[np.ndarray, np.ndarray]],
        d: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        alpha = np.asarray(alpha, dtype=int)
        if d is None:
            d = np.zeros((1, x.shape[1]), dtype=int)
        d = np.asarray(d, dtype=int)
        if alpha.shape[1] != x.shape[1] or d.shape[1] != x.shape[1]:
            raise ValueError("dimension mismatch")

        max_alpha = int(alpha.max(initial=0))
        a, b = recurrence_handle(max_alpha + 1)
        out = np.ones((x.shape[0], alpha.shape[0], d.shape[0]), dtype=float) / np.sqrt(b[0] ** x.shape[1])
        active_alpha = alpha > 0
        for qd in range(d.shape[0]):
            for qdim in range(x.shape[1]):
                temp = JacobiPolynomials.evaluate(a, b, x[:, qdim], int(alpha[:, qdim].max(initial=0)), int(d[qd, qdim]))
                if d[qd, qdim] > 0:
                    out[:, :, qd] *= temp[:, alpha[:, qdim]] * np.sqrt(b[0])
                else:
                    mask = active_alpha[:, qdim]
                    if np.any(mask):
                        out[:, mask, qd] *= temp[:, alpha[mask, qdim]] * np.sqrt(b[0])
        return out


def jacobi_recurrence(n: int, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    return JacobiPolynomials.recurrence(n, alpha, beta)


def legendre_recurrence(n: int) -> tuple[np.ndarray, np.ndarray]:
    return jacobi_recurrence(n, 0.0, 0.0)


def chebyshev_recurrence(n: int) -> tuple[np.ndarray, np.ndarray]:
    return jacobi_recurrence(n, -0.5, -0.5)


def poly_eval(a: np.ndarray, b: np.ndarray, x: np.ndarray, n: int, d: int = 0) -> np.ndarray:
    return JacobiPolynomials.evaluate(a, b, x, n, d)


def mpoly_eval(
    x: np.ndarray,
    alpha: np.ndarray,
    recurrence_handle: Callable[[int], tuple[np.ndarray, np.ndarray]],
    d: np.ndarray | None = None,
) -> np.ndarray:
    out = JacobiPolynomials.tensor_evaluate(x, alpha, recurrence_handle, d)
    return out[:, :, 0] if out.shape[2] == 1 else out


def chebyshev_eval(x: np.ndarray, alpha: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
    return mpoly_eval(np.asarray(x, dtype=float), np.asarray(alpha, dtype=int), chebyshev_recurrence, d)


def ratio_eval(a: np.ndarray, b: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    p = poly_eval(a, b, x, n)
    denom = np.where(np.abs(p[:, :-1]) > 0, p[:, :-1], np.inf)
    return p[:, 1:] / denom


def total_degree_indices(d: int, k: int) -> np.ndarray:
    rows: list[tuple[int, ...]] = []
    for total in range(k + 1):
        rows.extend(_compositions(total, d))
    return np.asarray(rows, dtype=int)


def _compositions(total: int, d: int) -> list[tuple[int, ...]]:
    if d == 1:
        return [(total,)]
    out: list[tuple[int, ...]] = []
    for first in range(total, -1, -1):
        for rest in _compositions(total - first, d - 1):
            out.append((first, *rest))
    return out


def hyperbolic_cross_indices(d: int, k: int) -> np.ndarray:
    if d < 1 or k < 0:
        raise ValueError("d >= 1 and k >= 0 required")
    max_degree = max(k, 0)
    grids = np.indices((max_degree + 1,) * d).reshape(d, -1).T
    keep = np.prod(grids + 1, axis=1) <= (k + 1)
    out = grids[keep]
    order = np.lexsort(tuple(out[:, j] for j in range(d - 1, -1, -1)))
    return out[order]


@dataclass
class PolynomialBasis:
    index_set: np.ndarray
    family: str = "legendre"
    alpha: float = 0.0
    beta: float = 0.0
    center: np.ndarray | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        self.index_set = np.asarray(self.index_set, dtype=int)
        if self.index_set.ndim != 2:
            raise ValueError("index_set must be 2D")
        self.family = self.family.lower()
        self.dimension = int(self.index_set.shape[1])
        if self.center is None:
            self.center = np.zeros(self.dimension, dtype=float)
        else:
            self.center = np.asarray(self.center, dtype=float).reshape(-1)
        if self.center.size != self.dimension:
            raise ValueError("center dimension mismatch")
        if self.family == "legendre":
            self.alpha = 0.0
            self.beta = 0.0
        elif self.family == "chebyshev":
            self.alpha = -0.5
            self.beta = -0.5

    @classmethod
    def from_total_degree(cls, dim: int, degree: int, **kwargs: object) -> "PolynomialBasis":
        return cls(total_degree_indices(dim, degree), **kwargs)

    @classmethod
    def from_hyperbolic_cross(cls, dim: int, degree: int, **kwargs: object) -> "PolynomialBasis":
        return cls(hyperbolic_cross_indices(dim, degree), **kwargs)

    def set_normalization(self, center: np.ndarray, scale: float) -> None:
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.size != self.dimension:
            raise ValueError("center dimension mismatch")
        self.center = center
        self.scale = float(scale)

    def fit_normalization_from_points(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        center = x.mean(axis=0)
        scale = np.linalg.norm(x - center, axis=1).max(initial=1.0)
        self.center = center
        self.scale = float(scale if scale > 0 else 1.0)

    def normalize_points(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.center) / self.scale

    def denormalize_points(self, xn: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(xn, dtype=float) + self.center

    def get_recurrence(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if self.family == "legendre":
            return legendre_recurrence(n)
        if self.family == "jacobi":
            return jacobi_recurrence(n, self.alpha, self.beta)
        if self.family == "chebyshev":
            return chebyshev_recurrence(n)
        raise ValueError(f"unknown family {self.family}")

    def evaluate(self, x: np.ndarray, d: np.ndarray | None = None, assume_normalized: bool = False) -> np.ndarray:
        if d is None:
            d = np.zeros((1, self.dimension), dtype=int)
        d = np.atleast_2d(np.asarray(d, dtype=int))
        xwork = np.asarray(x, dtype=float) if assume_normalized else self.normalize_points(x)
        p = JacobiPolynomials.tensor_evaluate(xwork, self.index_set, self.get_recurrence, d)
        orders = d.sum(axis=1)
        for q, order in enumerate(orders):
            if order > 0:
                p[:, :, q] /= self.scale**order
        return p[:, :, 0] if p.shape[2] == 1 else p
