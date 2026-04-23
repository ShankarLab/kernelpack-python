import numpy as np

from kernelpack import poly


def test_poly_basics():
    a, b = poly.jacobi_recurrence(5, 0, 0)
    al, bl = poly.legendre_recurrence(5)
    x = np.array([-1, -0.5, 0, 0.5, 1], dtype=float)
    p = poly.poly_eval(a, b, x, 3)
    dp = poly.poly_eval(a, b, x, 3, 1)
    assert np.allclose(a, al)
    assert np.allclose(b, bl)
    assert np.allclose(p[:, 0], 1 / np.sqrt(2))
    assert np.allclose(p[:, 1], np.sqrt(3 / 2) * x)
    assert np.allclose(dp[:, 1], np.sqrt(3 / 2))


def test_indices_and_basis():
    td = poly.total_degree_indices(2, 2)
    assert np.array_equal(td, np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]))
    hc = poly.hyperbolic_cross_indices(2, 3)
    assert any(np.all(row == np.array([0, 0])) for row in hc)
    assert any(np.all(row == np.array([1, 1])) for row in hc)
    assert not any(np.all(row == np.array([3, 3])) for row in hc)
    basis = poly.PolynomialBasis.from_total_degree(2, 2)
    basis.fit_normalization_from_points(np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float))
    assert np.linalg.norm(basis.center) < 1e-12
    assert abs(basis.scale - 1) < 1e-12
