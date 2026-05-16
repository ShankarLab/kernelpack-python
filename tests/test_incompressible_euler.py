import numpy as np

from kernelpack import geometry, nodes, solvers
from kernelpack.rbffd import StencilProperties


def build_dual_disk_domain():
    t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    curve = np.column_stack([np.cos(t), np.sin(t)])
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, 0.08, curve.shape[0])
    surface.build_level_set_from_geometric_model()

    generator = nodes.DualNodeDomainGenerator()
    generator.generate_smooth_domain_nodes(surface, 0.14, 0.18, seed=17, strip_count=5)
    return generator.create_dual_node_domain_descriptor()


def test_incompressible_euler_rotational_field_smoke():
    dual = build_dual_disk_domain()
    dim = dual.get_dim()
    velocity_sp = StencilProperties.from_accuracy(
        operator="grad",
        convergence_order=2,
        dimension=dim,
        approximation="rbf",
        tree_mode="interior_boundary",
        point_set="interior_boundary",
    )
    pressure_sp = StencilProperties.from_accuracy(
        operator="grad",
        convergence_order=2,
        dimension=dim,
        approximation="rbf",
        tree_mode="interior_boundary",
        point_set="interior_boundary",
    )

    solver = solvers.IncompressibleEulerSolver()
    solver.init(dual, velocity_sp, pressure_sp, 0.02)
    X = solver.Xphys
    u0 = np.column_stack([-X[:, 1], X[:, 0]])
    solver.set_initial_velocity(u0)
    walls = [solvers.IncompressibleEulerSolver.stationary_slip_wall(np.arange(1, solver.Xb.shape[0] + 1))]
    sol = solver.bdf1_step(lambda Xq: np.zeros_like(Xq), {"slip_walls": walls})

    assert sol["velocity"].shape == u0.shape
    assert sol["pressure"].shape[0] == solver.Xp.shape[0]
    assert np.all(np.isfinite(sol["velocity"]))
    assert np.all(np.isfinite(sol["pressure"]))
    assert sol["divergence_rms"] < 5e-1
    assert sol["wall_normal_rms"] < 5e-1
