import numpy as np

from kernelpack import geometry, nodes, solvers
from kernelpack.rbffd import StencilProperties
from kernelpack.solvers.detail import IncompressibleEulerBDFBackend


def build_dual_disk_domain():
    t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    curve = np.column_stack([np.cos(t), np.sin(t)])
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, 0.08, curve.shape[0])
    surface.build_level_set_from_geometric_model()

    generator = nodes.DualNodeDomainGenerator()
    generator.generate_smooth_domain_nodes_auto_pressure(
        surface,
        0.14,
        0.45,
        21,
        seed=17,
        strip_count=5,
        do_outer_refinement=True,
        outer_fraction_of_h=0.75,
        outer_refinement_zone_size_as_multiple_of_h=2.0,
    )
    return generator.create_dual_node_domain_descriptor()


def build_stencil_properties(dim: int) -> tuple[StencilProperties, StencilProperties]:
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
    return velocity_sp, pressure_sp


def rotational_velocity(x: np.ndarray) -> np.ndarray:
    return np.column_stack([-x[:, 1], x[:, 0]])


def test_pusl_incompressible_euler_smoke():
    dual = build_dual_disk_domain()
    velocity_sp, pressure_sp = build_stencil_properties(dual.get_dim())

    solver = solvers.PUSLIncompressibleEulerSolver()
    solver.init(dual, 4, velocity_sp, pressure_sp, 0.02)
    solver.set_tangential_flow_boundary(1.0e-6)
    u0 = rotational_velocity(solver.get_output_nodes())
    solver.set_initial_velocity(u0)

    wall = IncompressibleEulerBDFBackend.stationary_slip_wall(np.arange(1, solver.backend.xb.shape[0] + 1))
    problem = IncompressibleEulerBDFBackend.default_problem_definition()
    problem["slip_walls"] = [wall]
    sol = solver.bdf1_step(0.02, None, lambda time, x: np.zeros_like(x), problem)

    assert sol["velocity"].shape == u0.shape
    assert sol["pressure"].shape[0] == solver.backend.xp.shape[0]
    assert np.all(np.isfinite(sol["velocity"]))
    assert np.all(np.isfinite(sol["pressure"]))
    assert sol["divergence_rms"] < 7.5e-1
    assert sol["wall_normal_rms"] < 7.5e-1
