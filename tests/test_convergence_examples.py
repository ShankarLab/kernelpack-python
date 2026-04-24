from examples.poisson_convergence_2d_neumann import run_convergence_study
from examples.poisson_convergence_3d_neumann import run_convergence_study as run_3d_convergence_study


def test_poisson_neumann_convergence_study_smoke():
    results = run_convergence_study(
        "rbf",
        orders=(2,),
        h_values=(0.14, 0.10),
        curve_site_count=120,
    )
    assert results["backend"] == "rbf"
    assert results["orders"] == [2]
    assert len(results["rows"]) == 1
    assert len(results["rows"][0]) == 2
    assert len(results["rates"]) == 1
    assert len(results["rates"][0]["linf"]) == 1
    assert results["rows"][0][0]["linf"] > 0.0


def test_poisson_neumann_3d_convergence_study_smoke():
    results = run_3d_convergence_study(
        "rbf",
        orders=(2,),
        h_values=(0.28, 0.22),
        surface_site_count=120,
    )
    assert results["backend"] == "rbf"
    assert results["orders"] == [2]
    assert len(results["rows"]) == 1
    assert len(results["rows"][0]) == 2
    assert len(results["rates"]) == 1
    assert len(results["rates"][0]["linf"]) == 1
    assert results["rows"][0][0]["linf"] > 0.0
