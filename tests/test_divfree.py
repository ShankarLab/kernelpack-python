import numpy as np

from kernelpack import divfree


def _rotational_field(x: np.ndarray) -> np.ndarray:
    return np.column_stack([-x[:, 1], x[:, 0]])


def test_divfree_global_interpolant_reproduces_nodal_values():
    t = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
    x = np.column_stack([0.85 * np.cos(t), 0.65 * np.sin(t)])
    u = _rotational_field(x)

    interp = divfree.DivFreePHSInterpolant.fit(x, u, polyDegree=1, phsDegree=5)
    uhat = interp.evaluate(x)

    assert np.max(np.abs(uhat - u)) < 1e-9


def test_divfree_polynomial_basis_shape_and_eval():
    x = np.array(
        [
            [-0.8, -0.1],
            [-0.2, 0.4],
            [0.3, -0.5],
            [0.9, 0.2],
        ]
    )
    P, pdata = divfree.df_poly_basis_from_jacobi(x, 1)
    Pq = pdata.eval(x)

    assert P.shape[0] == 2 * x.shape[0]
    assert Pq.shape[0] == 2 * x.shape[0]
    assert P.shape[1] == Pq.shape[1]
    assert P.shape[1] > 0


def test_local_divfree_interpolator_reproduces_nodal_values():
    t = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    x = np.column_stack([np.cos(t), 0.75 * np.sin(t)])
    u = _rotational_field(x)

    interp = divfree.LocalDivFreeInterpolator.fit(
        x,
        u,
        polyDegree=1,
        phsDegree=5,
        stencilSize=8,
    )
    center_idx = np.arange(x.shape[0], dtype=int)
    uhat = interp.evaluate(x, CenterIndices=center_idx)

    assert np.max(np.abs(uhat - u)) < 1e-8


def test_local_divfree_interpolator_smoke_3d():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    u = np.column_stack([-pts[:, 1], pts[:, 0], np.zeros(pts.shape[0])])
    interp = divfree.LocalDivFreeInterpolator.fit(pts, u, polyDegree=1, phsDegree=5, stencilSize=8)
    q = np.array([[0.25, 0.25, 0.25], [0.5, 0.1, 0.75]], dtype=float)
    uq = interp.evaluate(q)

    assert uq.shape == q.shape
    assert np.all(np.isfinite(uq))
