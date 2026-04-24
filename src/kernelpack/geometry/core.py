from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy.spatial import Delaunay, cKDTree


def distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    y_sq = np.sum(y * y, axis=1, keepdims=True).T
    sq_dist = np.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)
    return np.sqrt(sq_dist)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return x / norms


def phs_kernel(r: np.ndarray, degree: int) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if degree % 2 == 0:
        return np.where(r > 0, r**degree * np.log(r + 2e-16), 0.0)
    return r**degree


def wrap_periodic_parameter(t: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(t, dtype=float), 1.0)


def periodic_chord_distance(theta1: np.ndarray, theta2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta1 = np.asarray(theta1, dtype=float).reshape(-1, 1)
    theta2 = np.asarray(theta2, dtype=float).reshape(1, -1)
    delta = theta1 - theta2
    return np.sqrt(2.0 - 2.0 * np.cos(delta)), delta


def sphere_chord_distance(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = normalize_rows(np.asarray(x, dtype=float))
    y = normalize_rows(np.asarray(y, dtype=float))
    dots = np.clip(x @ y.T, -1.0, 1.0)
    return np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0)), dots


def chord_length_param(x: np.ndarray, closed: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape[0] <= 1:
        return np.zeros(x.shape[0], dtype=float)
    diffs = np.diff(x, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    if closed:
        seg = np.concatenate([seg, [np.linalg.norm(x[0] - x[-1])]])
        s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
        return s / seg.sum()
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s / max(s[-1], 1.0)


def cart2sph_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    hxy = np.hypot(x[:, 0], x[:, 1])
    az = np.arctan2(x[:, 1], x[:, 0])
    el = np.arctan2(x[:, 2], hxy)
    r = np.linalg.norm(x, axis=1)
    return np.column_stack([az, el, r])


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=float)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def pca_oriented_bounding_box(x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    center = x.mean(axis=0)
    shifted = x - center
    _, _, vh = np.linalg.svd(shifted, full_matrices=False)
    local = shifted @ vh.T
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    corners_local = np.array(np.meshgrid(*zip(mins, maxs))).T.reshape(-1, x.shape[1])
    corners = corners_local @ vh + center
    return {"p": corners, "V": vh.T, "D": maxs - mins}


def weighted_sample_elimination_mis(x: np.ndarray, radius: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    tree = cKDTree(x)
    keep = np.zeros(x.shape[0], dtype=bool)
    blocked = np.zeros(x.shape[0], dtype=bool)
    for i in range(x.shape[0]):
        if blocked[i]:
            continue
        keep[i] = True
        for j in tree.query_ball_point(x[i], radius * (1 - 1e-12)):
            if j != i:
                blocked[j] = True
    return keep


def _connected_components_within_radius(x: np.ndarray, radius: float) -> list[np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.shape[0] == 0:
        return []
    tree = cKDTree(x)
    visited = np.zeros(x.shape[0], dtype=bool)
    components: list[np.ndarray] = []
    for i in range(x.shape[0]):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        component: list[int] = []
        while stack:
            j = stack.pop()
            component.append(j)
            for nbr in tree.query_ball_point(x[j], radius):
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)
        components.append(np.asarray(component, dtype=int))
    return components


def resample_closed_curve_by_arc_length(curve: np.ndarray, target_count: int) -> np.ndarray:
    # The Matlab port now chooses 2D closed-curve samples by arc length rather
    # than a generic MIS downsampling pass. This keeps neighboring boundary
    # samples more uniform around smooth curves.
    curve = np.asarray(curve, dtype=float)
    n = curve.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.uint32)
    if n == 1 or target_count <= 1:
        return np.array([0], dtype=np.uint32)

    shifted = np.vstack([curve[1:], curve[:1]])
    seg_lens = np.linalg.norm(shifted - curve, axis=1)
    total = float(seg_lens.sum())
    if total <= np.finfo(float).eps:
        count = min(n, target_count)
        return np.unique(np.round(np.linspace(0, n - 1, count)).astype(np.uint32))

    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    targets = np.arange(target_count, dtype=float) * (total / target_count)
    inds = np.searchsorted(cum_len, targets, side="right") - 1
    inds = np.clip(inds, 0, n - 1)
    next_inds = (inds + 1) % n
    current_dist = np.abs(cum_len[inds] - targets)
    next_dist = np.where(inds < n - 1, np.abs(cum_len[inds + 1] - targets), np.abs(total - targets))
    inds = np.where(next_dist < current_dist, next_inds, inds).astype(np.uint32)
    return np.unique(inds)


def project_to_best_fit_plane(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    origin = x.mean(axis=0)
    shifted = x - origin
    _, _, vh = np.linalg.svd(shifted, full_matrices=False)
    basis = vh[:2, :].T
    uv = shifted @ basis
    return uv, origin, basis


def points_in_convex_hull_2d(points: np.ndarray, hull_points: np.ndarray) -> np.ndarray:
    tri = Delaunay(np.asarray(hull_points, dtype=float))
    return tri.find_simplex(np.asarray(points, dtype=float)) >= 0


def build_planar_parametric_nodes_2d(x: np.ndarray, n: int, *_args: object) -> np.ndarray:
    uv, _, _ = project_to_best_fit_plane(np.asarray(x, dtype=float))
    if uv.shape[0] <= n:
        return uv
    keep = weighted_sample_elimination_mis(uv, 0.0)
    if keep.sum() < n:
        keep = np.zeros(uv.shape[0], dtype=bool)
        keep[np.linspace(0, uv.shape[0] - 1, n, dtype=int)] = True
    return uv[keep]


def build_planar_parametric_eval_nodes_2d(x: np.ndarray, n: int) -> np.ndarray:
    uv, _, _ = project_to_best_fit_plane(np.asarray(x, dtype=float))
    mins = uv.min(axis=0)
    maxs = uv.max(axis=0)
    side = int(np.ceil(np.sqrt(max(n, 1))))
    gx, gy = np.meshgrid(np.linspace(mins[0], maxs[0], side), np.linspace(mins[1], maxs[1], side))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    mask = points_in_convex_hull_2d(pts, uv)
    return pts[mask]


@dataclass
class RBFLevelSet:
    n: int = 0
    ell: int = 1
    dim: int = 0
    m_spline_degree: int = 3
    npoly: int = 0
    xd: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    nrd: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    ls_xd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mean_potential: float = 0.0
    centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    poly_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def build_level_set_from_cfi(self, x: np.ndarray, nr: np.ndarray, spline_degree: int = 3) -> None:
        # Construct a compact signed-distance surrogate from boundary points and
        # normals using the usual CFI-style inside/outside offset constraints.
        x = np.asarray(x, dtype=float)
        nr = normalize_rows(np.asarray(nr, dtype=float))
        self.dim = x.shape[1]
        self.n = x.shape[0]
        self.m_spline_degree = spline_degree
        self.npoly = self.dim + 1
        self.xd = x
        self.nrd = nr
        self.ls_xd = np.zeros(self.n, dtype=float)
        sep = self._estimate_offset_distance(x)
        inside = x - sep * nr
        outside = x + sep * nr
        self.centers = np.vstack([x, inside, outside])
        self.values = np.concatenate([np.zeros(self.n), -sep * np.ones(self.n), sep * np.ones(self.n)])
        nc = self.centers.shape[0]
        p = np.column_stack([np.ones(nc), self.centers])
        r = distance_matrix(self.centers, self.centers)
        k = phs_kernel(r, self.m_spline_degree)
        reg = 1e-12 * max(1.0, np.abs(k).max(initial=0.0))
        a = np.block([[k + reg * np.eye(nc), p], [p.T, np.zeros((self.dim + 1, self.dim + 1))]])
        rhs = np.concatenate([self.values, np.zeros(self.dim + 1)])
        coeffs = np.linalg.solve(a, rhs)
        self.weights = coeffs[:nc]
        self.poly_coeffs = coeffs[nc:]
        self.mean_potential = float(self.evaluate(x).mean())

    def evaluate(self, xe: np.ndarray) -> np.ndarray:
        model = self.get_evaluation_model()
        return self.evaluate_model(model, xe)

    def evaluate_gradient(self, xe: np.ndarray) -> np.ndarray:
        # Differentiate the RBF interpolant analytically with respect to each
        # physical coordinate.
        xe = np.asarray(xe, dtype=float)
        r = distance_matrix(xe, self.centers)
        dphi = self.m_spline_degree * r ** max(self.m_spline_degree - 2, 0)
        invr = np.divide(1.0, r, out=np.zeros_like(r), where=r > 0)
        grad = np.zeros((xe.shape[0], self.dim), dtype=float)
        for d in range(self.dim):
            delta = xe[:, d : d + 1] - self.centers[None, :, d]
            grad[:, d] = ((dphi * delta * invr) @ self.weights) + self.poly_coeffs[d + 1]
        return grad

    def project_to_surface_newton(self, initial_points: np.ndarray, options: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        opts = {
            "value_tolerance": 1e-12,
            "step_tolerance": 1e-12,
            "gradient_tolerance": 1e-14,
            "max_step_norm": np.inf,
            "max_iterations": 20,
        }
        if options:
            opts.update(options)
        x = np.asarray(initial_points, dtype=float).copy()
        converged = np.zeros(x.shape[0], dtype=bool)
        stalled = np.zeros(x.shape[0], dtype=bool)
        iterations = np.zeros(x.shape[0], dtype=int)
        for it in range(1, int(opts["max_iterations"]) + 1):
            active = ~(converged | stalled)
            if not np.any(active):
                break
            xa = x[active]
            phi = self.evaluate(xa)
            grad = self.evaluate_gradient(xa)
            g2 = np.sum(grad * grad, axis=1)
            indices = np.flatnonzero(active)
            for j, idx in enumerate(indices):
                iterations[idx] = it
                if abs(phi[j]) <= opts["value_tolerance"]:
                    converged[idx] = True
                    continue
                if g2[j] <= opts["gradient_tolerance"] ** 2:
                    stalled[idx] = True
                    continue
                step = -(phi[j] / g2[j]) * grad[j]
                step_norm = np.linalg.norm(step)
                if np.isfinite(opts["max_step_norm"]) and step_norm > opts["max_step_norm"]:
                    step *= opts["max_step_norm"] / step_norm
                x[idx] += step
                if np.linalg.norm(step) <= opts["step_tolerance"]:
                    converged[idx] = True
        final_phi = self.evaluate(x)
        return {
            "points": x,
            "level_set_values": final_phi,
            "iterations": iterations,
            "converged": np.abs(final_phi) <= opts["value_tolerance"],
            "stalled": stalled,
        }

    def is_point_in_surface(self, xe: np.ndarray, tol: float = 1e-3) -> np.ndarray:
        return (self.evaluate(xe) >= 0.5 * tol).astype(np.uint32)

    def is_point_outside_surface(self, xe: np.ndarray, tol: float = 1e-3) -> np.ndarray:
        return (self.evaluate(xe) <= -0.5 * tol).astype(np.uint32)

    def get_evaluation_model(self) -> dict[str, np.ndarray | float | int]:
        return {
            "centers": self.centers,
            "weights": self.weights,
            "poly_coeffs": self.poly_coeffs,
            "m_spline_degree": self.m_spline_degree,
            "mean_potential": self.mean_potential,
        }

    @staticmethod
    def evaluate_model(model: dict[str, np.ndarray | float | int], xe: np.ndarray) -> np.ndarray:
        xe = np.asarray(xe, dtype=float)
        centers = np.asarray(model["centers"], dtype=float)
        weights = np.asarray(model["weights"], dtype=float)
        poly_coeffs = np.asarray(model["poly_coeffs"], dtype=float)
        degree = int(model["m_spline_degree"])
        mean_potential = float(model["mean_potential"])
        r = distance_matrix(xe, centers)
        return phs_kernel(r, degree) @ weights + np.column_stack([np.ones(xe.shape[0]), xe]) @ poly_coeffs - mean_potential

    @staticmethod
    def _estimate_offset_distance(x: np.ndarray) -> float:
        if x.shape[0] < 2:
            return 1e-2
        distances, _ = cKDTree(x).query(x, k=2)
        nearest = np.asarray(distances[:, 1], dtype=float)
        return max(0.5 * float(nearest.min(initial=np.inf)), 1e-3)


@dataclass
class EmbeddedSurface:
    data_sites: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    data_site_nrmls: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    uniform_sample_sites: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    sample_sites: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    sample_sites_s: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    uniform_nrmls: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    nrmls: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    geom_model: dict[str, object] = field(default_factory=dict)
    nd: int = 0
    n: int = 0
    surf_dim: int = 0
    sep_rad: float = np.nan
    level_set: RBFLevelSet = field(default_factory=RBFLevelSet)
    tangent_step: float = 1e-5
    cbox: dict[str, np.ndarray] = field(default_factory=dict)
    ubox: dict[str, np.ndarray] = field(default_factory=dict)

    def set_data_sites(self, data_sites: np.ndarray) -> None:
        self.data_sites = np.asarray(data_sites, dtype=float)
        self.nd = self.data_sites.shape[0]
        self.surf_dim = self.data_sites.shape[1] - 1

    def set_sample_sites(self, sample_sites: np.ndarray) -> None:
        self.sample_sites = np.asarray(sample_sites, dtype=float)
        self.n = self.sample_sites.shape[0]

    def build_closed_geometric_model_ps(self, dim: int, rad: float, nb: int, ne: int | None = None, method: int = 1, supersample_fac: int = 2) -> None:
        # Build the smooth closed geometry model and then choose an evaluation
        # sample set. The 2D and 3D cases share the same idea but use different
        # parameterizations and sampling strategies.
        self.sep_rad = rad
        self.nd = min(nb, self.data_sites.shape[0])
        data = self.data_sites[: self.nd]
        degree = 7
        ntarget = ne if ne is not None else self._estimate_evaluation_count(dim, rad, True)
        if dim == 2:
            t = chord_length_param(data, True)
            theta = 2.0 * np.pi * t
            r, _ = periodic_chord_distance(theta, theta)
            k = phs_kernel(r, degree)
            reg = 1e-12 * max(1.0, np.abs(k).max(initial=0.0))
            weights = np.linalg.solve(k + reg * np.eye(k.shape[0]), data)
            self.geom_model = {"type": "closed-curve-sbf", "degree": degree, "theta": theta, "weights": weights}
            _, dn = self._eval_closed_curve_frame(t)
            self.data_site_nrmls = dn
            if method == 1:
                # Oversample the smooth closed curve first, then resample by arc
                # length so the boundary cloud matches the newer Matlab behavior.
                ns = max(round(supersample_fac * 1.5 * ntarget), ntarget)
                ts = np.linspace(0.0, 1.0, ns, endpoint=False)
                ptss = self._eval_closed_curve(ts)
                tan, nr = self._eval_closed_curve_frame(ts)
                self.sample_sites_s = ptss
                curve_length = np.linalg.norm(np.vstack([ptss[1:], ptss[:1]]) - ptss, axis=1).sum()
                target_spacing = max(np.sqrt(2.0) * rad, np.finfo(float).eps)
                target_count = max(2, round(curve_length / target_spacing))
                keep_inds = resample_closed_curve_by_arc_length(ptss, target_count).astype(int)
                self.sample_sites = ptss[keep_inds]
                self.nrmls = nr[keep_inds]
            else:
                # The direct path evaluates exactly the requested number of
                # samples without any secondary resampling pass.
                ns = ntarget
                ts = np.linspace(0.0, 1.0, ns, endpoint=False)
                self.sample_sites = self._eval_closed_curve(ts)
                _, self.nrmls = self._eval_closed_curve_frame(ts)
            self.uniform_sample_sites = self.sample_sites
            self.uniform_nrmls = self.nrmls
        elif dim == 3:
            # Closed surfaces still use the older oversample-and-MIS pattern.
            center = data.mean(axis=0)
            local = data - center
            uvw = cart2sph_rows(local)
            unit_centers = local / np.maximum(uvw[:, 2:3], np.finfo(float).eps)
            r, _ = sphere_chord_distance(unit_centers, unit_centers)
            k = phs_kernel(r, degree)
            reg = 1e-12 * max(1.0, np.abs(k).max(initial=0.0))
            weights = np.linalg.solve(k + reg * np.eye(k.shape[0]), data)
            self.geom_model = {"type": "closed-surface-sbf", "degree": degree, "center": center, "unit_centers": unit_centers, "weights": weights}
            ns = max(round(supersample_fac * ntarget), ntarget) if method == 1 else ntarget
            xyz = fibonacci_sphere(ns)
            uv = cart2sph_rows(xyz)[:, :2]
            ptss = self._eval_closed_surface(uv)
            _, _, nr = self._eval_closed_surface_frame(uv)
            keep = weighted_sample_elimination_mis(ptss, rad) if method == 1 else np.ones(ptss.shape[0], dtype=bool)
            if keep.sum() < max(8, ntarget // 2):
                keep = np.zeros(ptss.shape[0], dtype=bool)
                keep[np.linspace(0, ptss.shape[0] - 1, ntarget, dtype=int)] = True
            self.sample_sites = ptss[keep]
            self.nrmls = nr[keep]
            self.uniform_sample_sites = self.sample_sites
            self.uniform_nrmls = self.nrmls
        else:
            raise ValueError("unsupported dimension")
        self.n = self.sample_sites.shape[0]

    def build_geometric_model_ps(self, dim: int, rad: float, nb: int, ne: int | None = None, method: int = 1, supersample_fac: int = 2) -> None:
        self.sep_rad = rad
        self.nd = min(nb, self.data_sites.shape[0])
        data = self.data_sites[: self.nd]
        degree = 7
        ntarget = ne if ne is not None else self._estimate_evaluation_count(dim, rad, False)
        if dim == 2:
            u = chord_length_param(data, False)
            r = distance_matrix(u[:, None], u[:, None])
            k = phs_kernel(r, degree)
            p = np.column_stack([np.ones(u.size), u])
            reg = 1e-12 * max(1.0, np.abs(k).max(initial=0.0))
            a = np.block([[k + reg * np.eye(u.size), p], [p.T, np.zeros((2, 2))]])
            coeffs = np.linalg.solve(a, np.vstack([data, np.zeros((2, data.shape[1]))]))
            self.geom_model = {
                "type": "open-curve-rbf",
                "degree": degree,
                "u": u,
                "rbf_weights": coeffs[: u.size],
                "poly_coeffs": coeffs[u.size :],
            }
            us = np.linspace(0.0, 1.0, max(round(supersample_fac * 1.5 * ntarget), ntarget)) if method == 1 else np.linspace(0.0, 1.0, ntarget)
            pts = self._eval_open_curve(us)
            _, nr = self._eval_open_curve_frame(us)
            keep = weighted_sample_elimination_mis(pts, rad) if method == 1 else np.ones(pts.shape[0], dtype=bool)
            if not np.any(keep):
                keep[0] = True
            self.sample_sites = pts[keep]
            self.nrmls = nr[keep]
            self.uniform_sample_sites = self.sample_sites
            self.uniform_nrmls = self.nrmls
        elif dim == 3:
            uv = build_planar_parametric_nodes_2d(data, self.nd)
            r = distance_matrix(uv, uv)
            k = phs_kernel(r, degree)
            p = np.column_stack([np.ones(uv.shape[0]), uv])
            reg = 1e-12 * max(1.0, np.abs(k).max(initial=0.0))
            a = np.block([[k + reg * np.eye(uv.shape[0]), p], [p.T, np.zeros((3, 3))]])
            coeffs = np.linalg.solve(a, np.vstack([data, np.zeros((3, data.shape[1]))]))
            _, origin, basis = project_to_best_fit_plane(data)
            self.geom_model = {
                "type": "surface-patch-rbf",
                "degree": degree,
                "uv": uv,
                "rbf_weights": coeffs[: uv.shape[0]],
                "poly_coeffs": coeffs[uv.shape[0] :],
                "origin": origin,
                "basis": basis,
            }
            uv_eval = build_planar_parametric_eval_nodes_2d(data, ntarget)
            pts = self._eval_open_surface(uv_eval)
            _, _, nr = self._eval_open_surface_frame(uv_eval)
            self.sample_sites = pts
            self.nrmls = nr
            self.uniform_sample_sites = self.sample_sites
            self.uniform_nrmls = self.nrmls
        else:
            raise ValueError("unsupported dimension")
        self.n = self.sample_sites.shape[0]

    def build_level_set_from_geometric_model(self, lambdas: np.ndarray | None = None) -> None:
        if lambdas is None or len(np.atleast_1d(lambdas)) == 0:
            pts = self.uniform_sample_sites
            nr = self.uniform_nrmls
        else:
            vals = np.asarray(lambdas, dtype=float)
            if self.geom_model["type"] == "closed-curve-sbf":
                pts = self._eval_closed_curve(vals)
                _, nr = self._eval_closed_curve_frame(vals)
            elif self.geom_model["type"] == "closed-surface-sbf":
                pts = self._eval_closed_surface(vals)
                _, _, nr = self._eval_closed_surface_frame(vals)
            elif self.surf_dim == 1:
                pts = self._eval_open_curve(vals)
                _, nr = self._eval_open_curve_frame(vals)
            else:
                pts = self._eval_open_surface(vals)
                _, _, nr = self._eval_open_surface_frame(vals)
        self.level_set = RBFLevelSet()
        self.level_set.build_level_set_from_cfi(pts, nr)

    def compute_bounding_box(self) -> None:
        self.cbox = pca_oriented_bounding_box(self.sample_sites)

    def compute_uniform_bounding_box(self) -> None:
        self.ubox = pca_oriented_bounding_box(self.uniform_sample_sites)

    def flip_normals(self) -> None:
        self.nrmls = -self.nrmls
        self.uniform_nrmls = -self.uniform_nrmls
        self.data_site_nrmls = -self.data_site_nrmls

    def get_sample_sites(self) -> np.ndarray:
        return self.sample_sites

    def get_uniform_sample_sites(self) -> np.ndarray:
        return self.uniform_sample_sites

    def get_nrmls(self) -> np.ndarray:
        return self.nrmls

    def get_uniform_nrmls(self) -> np.ndarray:
        return self.uniform_nrmls

    def get_bounding_box(self) -> np.ndarray:
        return self.cbox.get("p", np.zeros((0, self.sample_sites.shape[1] if self.sample_sites.size else 0)))

    def get_uniform_bounding_box(self) -> np.ndarray:
        return self.ubox.get("p", np.zeros((0, self.sample_sites.shape[1] if self.sample_sites.size else 0)))

    def get_level_set(self) -> RBFLevelSet:
        return self.level_set

    def get_n(self) -> int:
        return self.n

    def _estimate_evaluation_count(self, dim: int, rad: float, closed: bool) -> int:
        x = self.data_sites[: self.nd]
        if dim == 2:
            if x.shape[0] < 2:
                return 8
            dx = np.diff(np.vstack([x, x[:1]]) if closed else x, axis=0)
            measure = np.linalg.norm(dx, axis=1).sum()
            return max(int(np.ceil(measure / max(rad, np.finfo(float).eps))), 8)
        mins = x.min(axis=0)
        maxs = x.max(axis=0)
        ext = np.maximum(maxs - mins, np.finfo(float).eps)
        area = 2.0 * (ext[0] * ext[1] + ext[0] * ext[2] + ext[1] * ext[2])
        if not closed:
            area *= 0.5
        return max(int(np.ceil(area / max(rad * rad, np.finfo(float).eps))), 16)

    def _eval_closed_curve(self, t: np.ndarray) -> np.ndarray:
        tw = wrap_periodic_parameter(np.asarray(t, dtype=float).reshape(-1))
        theta = 2.0 * np.pi * tw
        r, _ = periodic_chord_distance(theta, self.geom_model["theta"])
        return phs_kernel(r, int(self.geom_model["degree"])) @ np.asarray(self.geom_model["weights"], dtype=float)

    def _eval_closed_curve_frame(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tw = wrap_periodic_parameter(np.asarray(t, dtype=float).reshape(-1))
        theta = 2.0 * np.pi * tw
        r, delta = periodic_chord_distance(theta, self.geom_model["theta"])
        degree = int(self.geom_model["degree"])
        if degree == 1:
            dphi = np.divide(np.sin(delta), np.maximum(r, np.finfo(float).eps))
            dphi[r == 0] = 0.0
        else:
            dphi = degree * np.sin(delta) * r ** (degree - 2)
            dphi[r == 0] = 0.0
        xt = (2.0 * np.pi) * (dphi @ np.asarray(self.geom_model["weights"], dtype=float))
        n = normalize_rows(np.column_stack([xt[:, 1], -xt[:, 0]]))
        return xt, n

    def _eval_open_curve(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float).reshape(-1), 0.0, 1.0)
        r = distance_matrix(u[:, None], np.asarray(self.geom_model["u"], dtype=float)[:, None])
        return phs_kernel(r, int(self.geom_model["degree"])) @ np.asarray(self.geom_model["rbf_weights"], dtype=float) + np.column_stack([np.ones(u.size), u]) @ np.asarray(self.geom_model["poly_coeffs"], dtype=float)

    def _eval_open_curve_frame(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.clip(np.asarray(u, dtype=float).reshape(-1), 0.0, 1.0)
        du = u[:, None] - np.asarray(self.geom_model["u"], dtype=float)[None, :]
        r = np.abs(du)
        degree = int(self.geom_model["degree"])
        dphi = np.sign(du) if degree == 1 else degree * np.sign(du) * r ** (degree - 1)
        dphi[(r == 0) & (degree > 1)] = 0.0
        xt = dphi @ np.asarray(self.geom_model["rbf_weights"], dtype=float) + np.asarray(self.geom_model["poly_coeffs"], dtype=float)[1]
        n = normalize_rows(np.column_stack([xt[:, 1], -xt[:, 0]]))
        return xt, n

    def _eval_closed_surface(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=float)
        unit_query = np.column_stack([np.cos(uv[:, 1]) * np.cos(uv[:, 0]), np.cos(uv[:, 1]) * np.sin(uv[:, 0]), np.sin(uv[:, 1])])
        r, _ = sphere_chord_distance(unit_query, np.asarray(self.geom_model["unit_centers"], dtype=float))
        return phs_kernel(r, int(self.geom_model["degree"])) @ np.asarray(self.geom_model["weights"], dtype=float)

    def _eval_closed_surface_frame(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        uv = np.asarray(uv, dtype=float)
        h = self.tangent_step
        uv_up = uv.copy()
        uv_um = uv.copy()
        uv_up[:, 0] += h
        uv_um[:, 0] -= h
        tu = (self._eval_closed_surface(uv_up) - self._eval_closed_surface(uv_um)) / (2.0 * h)
        uv_vp = uv.copy()
        uv_vm = uv.copy()
        uv_vp[:, 1] = np.minimum(uv[:, 1] + h, np.pi / 2.0)
        uv_vm[:, 1] = np.maximum(uv[:, 1] - h, -np.pi / 2.0)
        denom = np.maximum((uv_vp[:, 1] - uv_vm[:, 1])[:, None], np.finfo(float).eps)
        tv = (self._eval_closed_surface(uv_vp) - self._eval_closed_surface(uv_vm)) / denom
        n = normalize_rows(np.cross(tu, tv))
        return tu, tv, n

    def _eval_open_surface(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=float)
        r = distance_matrix(uv, np.asarray(self.geom_model["uv"], dtype=float))
        return phs_kernel(r, int(self.geom_model["degree"])) @ np.asarray(self.geom_model["rbf_weights"], dtype=float) + np.column_stack([np.ones(uv.shape[0]), uv]) @ np.asarray(self.geom_model["poly_coeffs"], dtype=float)

    def _eval_open_surface_frame(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        uv = np.asarray(uv, dtype=float)
        centers = np.asarray(self.geom_model["uv"], dtype=float)
        du = uv[:, 0:1] - centers[None, :, 0]
        dv = uv[:, 1:2] - centers[None, :, 1]
        r = np.sqrt(du * du + dv * dv)
        degree = int(self.geom_model["degree"])
        dphi = np.ones_like(r) if degree == 1 else degree * r ** max(degree - 2, 0)
        dphi[(r == 0) & (degree > 1)] = 0.0
        weights = np.asarray(self.geom_model["rbf_weights"], dtype=float)
        poly = np.asarray(self.geom_model["poly_coeffs"], dtype=float)
        tu = (dphi * du) @ weights + poly[1]
        tv = (dphi * dv) @ weights + poly[2]
        n = normalize_rows(np.cross(tu, tv))
        return tu, tv, n


@dataclass
class PiecewiseSmoothEmbeddedSurface:
    segments: list[EmbeddedSurface] = field(default_factory=list)
    xb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xb_uniform: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    nrmls: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    nrmls_uniform: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    segment_map: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    corner_flags: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    level_set: RBFLevelSet = field(default_factory=RBFLevelSet)
    cbox: dict[str, np.ndarray] = field(default_factory=dict)
    ubox: dict[str, np.ndarray] = field(default_factory=dict)

    def generate_piecewise_smooth_surface_by_segment(
        self,
        bdry_segments: list[np.ndarray],
        flip_normal: list[bool] | np.ndarray,
        radius: float,
        method: int = 1,
        supersample_fac: int = 2,
        _mode: int = 2,
        smooth_normals: bool = False,
        smooth_neighborhood: int = 0,
    ) -> None:
        self.segments = []
        dim = np.asarray(bdry_segments[0]).shape[1]
        for seg_pts, flip in zip(bdry_segments, flip_normal):
            seg = EmbeddedSurface()
            seg.set_data_sites(np.asarray(seg_pts, dtype=float))
            seg.compute_bounding_box = lambda seg=seg: setattr(seg, "cbox", pca_oriented_bounding_box(seg.sample_sites if seg.sample_sites.size else seg.data_sites))
            seg.set_sample_sites(seg_pts)
            seg.compute_bounding_box()
            box = seg.get_bounding_box()
            ext = box.max(axis=0) - box.min(axis=0)
            if dim == 2:
                bdry_size = 2.0 * (ext[0] + ext[1])
                seg_n = max(8, round(bdry_size / radius))
            else:
                bdry_size = 2.0 * (ext[0] * ext[1] + ext[0] * ext[2] + ext[1] * ext[2])
                seg_n = max(16, round(bdry_size / max(radius * radius, np.finfo(float).eps)))
            seg.build_geometric_model_ps(dim, radius, len(seg_pts), seg_n, method, supersample_fac)
            if flip:
                seg.flip_normals()
            self.segments.append(seg)

        xb = []
        nr = []
        segmap = []
        corners = []
        xb_u = []
        nr_u = []
        segmap_u = []
        corners_u = []
        for k, seg in enumerate(self.segments, start=1):
            pts = seg.get_sample_sites()
            nrm = seg.get_nrmls()
            xb.append(pts)
            nr.append(nrm)
            segmap.extend([k] * pts.shape[0])
            flags = np.zeros(pts.shape[0])
            if pts.shape[0]:
                flags[0] = 1
                flags[-1] = 1
            corners.append(flags)
            ptsu = seg.get_uniform_sample_sites()
            nrmu = seg.get_uniform_nrmls()
            xb_u.append(ptsu)
            nr_u.append(nrmu)
            segmap_u.extend([k] * ptsu.shape[0])
            flagsu = np.zeros(ptsu.shape[0])
            if ptsu.shape[0]:
                flagsu[0] = 1
                flagsu[-1] = 1
            corners_u.append(flagsu)
        self.xb, self.nrmls, self.segment_map, self.corner_flags = self._deduplicate_boundary(
            np.vstack(xb), np.vstack(nr), np.asarray(segmap), np.concatenate(corners), 0.2 * radius
        )
        self.xb, self.nrmls, self.segment_map, self.corner_flags = self._enforce_minimum_spacing(
            self.xb, self.nrmls, self.segment_map, self.corner_flags, radius
        )
        self.xb_uniform, self.nrmls_uniform, _, _ = self._deduplicate_boundary(
            np.vstack(xb_u), np.vstack(nr_u), np.asarray(segmap_u), np.concatenate(corners_u), 0.2 * radius
        )
        self.xb_uniform, self.nrmls_uniform, _, _ = self._enforce_minimum_spacing(
            self.xb_uniform, self.nrmls_uniform, np.asarray(segmap_u[: self.xb_uniform.shape[0]]), np.zeros(self.xb_uniform.shape[0]), radius
        )
        if smooth_normals and smooth_neighborhood > 1:
            self.nrmls = self._smooth_normals(self.xb, self.nrmls, smooth_neighborhood)
            self.nrmls_uniform = self._smooth_normals(self.xb_uniform, self.nrmls_uniform, smooth_neighborhood)

    def build_level_set(self) -> None:
        self.level_set = RBFLevelSet()
        self.level_set.build_level_set_from_cfi(self.xb_uniform, self.nrmls_uniform)

    def compute_bounding_box(self) -> None:
        self.cbox = pca_oriented_bounding_box(self.xb)

    def compute_uniform_bounding_box(self) -> None:
        self.ubox = pca_oriented_bounding_box(self.xb_uniform)

    def get_level_set(self) -> RBFLevelSet:
        return self.level_set

    def get_bdry_nodes(self) -> np.ndarray:
        return self.xb

    def get_uniform_bdry_nodes(self) -> np.ndarray:
        return self.xb_uniform

    def get_bdry_nrmls(self) -> np.ndarray:
        return self.nrmls

    def get_uniform_bdry_nrmls(self) -> np.ndarray:
        return self.nrmls_uniform

    def get_corner_flags(self) -> np.ndarray:
        return self.corner_flags

    def get_sample_sites(self) -> np.ndarray:
        return self.xb

    def get_uniform_sample_sites(self) -> np.ndarray:
        return self.xb_uniform

    def get_nrmls(self) -> np.ndarray:
        return self.nrmls

    def get_uniform_nrmls(self) -> np.ndarray:
        return self.nrmls_uniform

    def get_bounding_box(self) -> np.ndarray:
        return self.cbox.get("p", np.zeros((0, self.xb.shape[1] if self.xb.size else 0)))

    def get_uniform_bounding_box(self) -> np.ndarray:
        return self.ubox.get("p", np.zeros((0, self.xb_uniform.shape[1] if self.xb_uniform.size else 0)))

    def _deduplicate_boundary(self, x: np.ndarray, n: np.ndarray, seg_map: np.ndarray, corner_flags: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_keep = []
        n_keep = []
        seg_keep = []
        corner_keep = []
        for cluster in _connected_components_within_radius(x, tol):
            x_keep.append(x[cluster].mean(axis=0))
            nr = n[cluster].copy()
            ref = nr[0]
            for j in range(1, nr.shape[0]):
                if np.dot(nr[j], ref) < 0:
                    nr[j] *= -1.0
            navg = normalize_rows(nr.mean(axis=0, keepdims=True))[0]
            x_keep[-1] = np.asarray(x_keep[-1])
            n_keep.append(navg)
            seg_keep.append(seg_map[cluster[0]])
            is_corner = np.any(corner_flags[cluster] != 0) or np.unique(seg_map[cluster]).size > 1
            if nr.shape[0] > 1:
                dots = nr @ nr.T
                upper = dots[np.triu_indices_from(dots, k=1)]
                if np.any(upper < np.cos(np.deg2rad(35.0))):
                    is_corner = True
            corner_keep.append(float(is_corner))
        return np.vstack(x_keep), np.vstack(n_keep), np.asarray(seg_keep), np.asarray(corner_keep)

    def _enforce_minimum_spacing(self, x: np.ndarray, n: np.ndarray, seg_map: np.ndarray, corner_flags: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        keep = np.ones(x.shape[0], dtype=bool)
        corner_mask = corner_flags != 0
        order = np.concatenate([np.flatnonzero(corner_mask), np.flatnonzero(~corner_mask)])
        spacing_tol = radius * (1.0 - 1e-12)
        tree = cKDTree(x)
        for idx in order:
            if not keep[idx]:
                continue
            nbrs = np.asarray(tree.query_ball_point(x[idx], spacing_tol), dtype=int)
            for j in nbrs:
                if j == idx:
                    continue
                if not keep[j]:
                    continue
                if corner_mask[j] and not corner_mask[idx]:
                    keep[idx] = False
                    break
                keep[j] = False
        return x[keep], n[keep], seg_map[keep], corner_flags[keep]

    def _smooth_normals(self, x: np.ndarray, nr: np.ndarray, neighborhood: int) -> np.ndarray:
        out = nr.copy()
        tree = cKDTree(x)
        for i in range(x.shape[0]):
            _, take = tree.query(x[i], k=min(neighborhood, x.shape[0]))
            take = np.atleast_1d(np.asarray(take, dtype=int))
            nri = nr[take].copy()
            ref = nri[0]
            for j in range(1, nri.shape[0]):
                if np.dot(nri[j], ref) < 0:
                    nri[j] *= -1.0
            out[i] = normalize_rows(nri.mean(axis=0, keepdims=True))[0]
        return out
