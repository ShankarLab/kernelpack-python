from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from kernelpack.geometry import distance_matrix


@dataclass
class TreeStruct:
    points: np.ndarray
    searcher: cKDTree | None
    has_searcher: bool


@dataclass
class DomainDescriptor:
    xi: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xg: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xf: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    nrmls: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    sep_rad: float = np.nan
    tall_tree: TreeStruct | None = None
    int_bdry_tree: TreeStruct | None = None
    bdry_tree: TreeStruct | None = None
    outer_level_set: object | None = None
    boundary_level_sets: list[object] = field(default_factory=list)

    def set_nodes(self, int_nodes: np.ndarray, bdry_nodes: np.ndarray, ghost_nodes: np.ndarray | None = None) -> None:
        # Normalize the three node groups into the internal storage layout used
        # everywhere else in the package. The descriptor keeps the physically
        # meaningful partitions (`xi`, `xb`, `xg`) as well as concatenated views
        # that make assembly/query code simpler.
        int_nodes = np.asarray(int_nodes, dtype=float)
        bdry_nodes = np.asarray(bdry_nodes, dtype=float)
        if ghost_nodes is None:
            ghost_nodes = np.zeros((0, int_nodes.shape[1] if int_nodes.size else bdry_nodes.shape[1]), dtype=float)
        self.xi = int_nodes
        self.xb = bdry_nodes
        self.xg = np.asarray(ghost_nodes, dtype=float)
        self._set_total_nodes()

    def set_normals(self, nrmls: np.ndarray) -> None:
        nrmls = np.asarray(nrmls, dtype=float)
        if nrmls.shape[0] != self.xb.shape[0]:
            raise ValueError("boundary normals must match boundary node count")
        self.nrmls = nrmls

    def set_outer_level_set(self, level_set: object) -> None:
        self.outer_level_set = level_set

    def set_sep_rad(self, sep_rad: float) -> None:
        self.sep_rad = float(sep_rad)

    def set_boundary_level_sets(self, level_sets: list[object]) -> None:
        self.boundary_level_sets = level_sets

    def build_structs(self) -> None:
        # Build the search structures once after the node sets are finalized.
        # Most downstream code asks the descriptor for one of three search
        # spaces: all nodes, interior+boundary nodes, or boundary nodes only.
        self.tall_tree = self._build_tree_struct(self.xf)
        self.int_bdry_tree = self._build_tree_struct(self.x)
        self.bdry_tree = self._build_tree_struct(self.xb)

    def query_knn(self, tree_mode: str, query_points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        tree, points = self._get_tree_data(tree_mode)
        query_points = np.asarray(query_points, dtype=float)
        query_points = np.atleast_2d(query_points)
        if points.size == 0:
            return np.zeros((query_points.shape[0], 0), dtype=int), np.zeros((query_points.shape[0], 0))
        k = min(int(k), points.shape[0])
        if tree and tree.has_searcher and tree.searcher is not None:
            # Use the KD-tree when available: this is the hot path for stencil
            # construction, so we keep the fallback path only for completeness.
            distances, indices = tree.searcher.query(query_points, k=k)
            indices = np.asarray(indices, dtype=int).reshape(query_points.shape[0], k)
            distances = np.asarray(distances, dtype=float).reshape(query_points.shape[0], k)
            return indices, distances
        # Fallback for degenerate cases where we deliberately did not build a
        # searcher. This keeps the descriptor usable even with tiny point sets.
        d = distance_matrix(query_points, points)
        order = np.argsort(d, axis=1)[:, :k]
        distances = np.take_along_axis(d, order, axis=1)
        return order, distances

    def query_ball(self, tree_mode: str, query_points: np.ndarray, radius: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        tree, points = self._get_tree_data(tree_mode)
        query_points = np.atleast_2d(np.asarray(query_points, dtype=float))
        if points.size == 0:
            return [np.zeros(0, dtype=int) for _ in range(query_points.shape[0])], [np.zeros(0) for _ in range(query_points.shape[0])]
        if tree and tree.has_searcher and tree.searcher is not None:
            raw_idx = tree.searcher.query_ball_point(query_points, radius)
            indices = [np.asarray(i, dtype=int) for i in raw_idx]
            distances = []
            for q, idx in zip(query_points, indices):
                if idx.size == 0:
                    distances.append(np.zeros(0))
                    continue
                diff = points[idx] - q
                distances.append(np.sqrt(np.sum(diff * diff, axis=1)))
            return indices, distances
        d = distance_matrix(query_points, points)
        indices = []
        distances = []
        for row in d:
            mask = row <= radius
            indices.append(np.flatnonzero(mask))
            distances.append(row[mask])
        return indices, distances

    def get_tree_points(self, tree_mode: str) -> np.ndarray:
        return self._get_tree_data(tree_mode)[1]

    def get_tree_globals(self, tree_mode: str) -> np.ndarray:
        tree_mode = _normalize_tree_mode(tree_mode)
        if tree_mode == "all":
            return np.arange(1, self.xf.shape[0] + 1)
        if tree_mode == "interior_boundary":
            return np.arange(1, self.x.shape[0] + 1)
        if tree_mode == "boundary":
            return self.get_num_interior_nodes() + np.arange(1, self.xb.shape[0] + 1)
        raise ValueError(f"unknown tree mode {tree_mode}")

    def get_interior_nodes(self) -> np.ndarray:
        return self.xi

    def get_bdry_nodes(self) -> np.ndarray:
        return self.xb

    def get_ghost_nodes(self) -> np.ndarray:
        return self.xg

    def get_int_bdry_nodes(self) -> np.ndarray:
        return self.x

    def get_all_nodes(self) -> np.ndarray:
        return self.xf

    def get_nrmls(self) -> np.ndarray:
        return self.nrmls

    def get_outer_level_set(self) -> object | None:
        return self.outer_level_set

    def get_boundary_level_sets(self) -> list[object]:
        return self.boundary_level_sets

    def get_sep_rad(self) -> float:
        return self.sep_rad

    def get_dim(self) -> int:
        return self.xf.shape[1]

    def get_num_total_nodes(self) -> int:
        return self.xf.shape[0]

    def get_num_int_bdry_nodes(self) -> int:
        return self.x.shape[0]

    def get_num_interior_nodes(self) -> int:
        return self.xi.shape[0]

    def get_num_bdry_nodes(self) -> int:
        return self.xb.shape[0]

    def _set_total_nodes(self) -> None:
        # Maintain the concatenated node arrays expected by the assembler code.
        # `x` is the physical point cloud (interior + boundary), while `xf`
        # appends ghost nodes for methods that need the full augmented domain.
        dim = self.xi.shape[1] if self.xi.size else self.xb.shape[1]
        if self.xg.size == 0:
            self.xg = np.zeros((0, dim))
        self.x = np.vstack([self.xi, self.xb]) if self.xb.size or self.xi.size else np.zeros((0, dim))
        self.xf = np.vstack([self.x, self.xg]) if self.xg.size or self.x.size else np.zeros((0, dim))

    def _get_tree_data(self, tree_mode: str) -> tuple[TreeStruct | None, np.ndarray]:
        tree_mode = _normalize_tree_mode(tree_mode)
        if tree_mode == "all":
            return self.tall_tree, self.xf
        if tree_mode == "interior_boundary":
            return self.int_bdry_tree, self.x
        if tree_mode == "boundary":
            return self.bdry_tree, self.xb
        raise ValueError(f"unknown tree mode {tree_mode}")

    @staticmethod
    def _build_tree_struct(points: np.ndarray) -> TreeStruct:
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return TreeStruct(points=points, searcher=None, has_searcher=False)
        # Cache a KD-tree directly on the descriptor so repeated stencil queries
        # do not rebuild spatial data structures.
        return TreeStruct(points=points, searcher=cKDTree(points), has_searcher=True)


def _normalize_tree_mode(mode: str) -> str:
    mode = str(mode).lower()
    aliases = {
        "all": "all",
        "all_nodes": "all",
        "full": "all",
        "interior_boundary": "interior_boundary",
        "interior+boundary": "interior_boundary",
        "int_bdry": "interior_boundary",
        "boundary": "boundary",
        "bdry": "boundary",
        "boundary_only": "boundary",
    }
    if mode not in aliases:
        raise ValueError(f"unknown tree mode {mode}")
    return aliases[mode]
