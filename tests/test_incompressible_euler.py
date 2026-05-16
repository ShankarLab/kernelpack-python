import numpy as np

from kernelpack import geometry, nodes, solvers
from kernelpack.rbffd import StencilProperties
from kernelpack.solvers.detail import IncompressibleEulerBDFBackend


def build_dual_disk_domain(h: float = 0.14, pressure_fraction: float = 0.45, min_pressure_nodes: int = 21):
    t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    curve = np.column_stack([np.cos(t), np.sin(t)])
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, h, curve.shape[0])
    surface.build_level_set_from_geometric_model()

    generator = nodes.DualNodeDomainGenerator()
    generator.generate_smooth_domain_nodes_auto_pressure(
        surface,
        h,
        pressure_fraction,
        min_pressure_nodes,
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


def amplitude(t: float) -> float:
    return 1.0 + 0.2 * np.sin(2.0 * t) + 0.15 * np.cos(7.0 * t)


def amplitude_dt(t: float) -> float:
    return 0.4 * np.cos(2.0 * t) - 1.05 * np.sin(7.0 * t)


def velocity_shape(x: np.ndarray) -> np.ndarray:
    x0 = x[:, 0]
    x1 = x[:, 1]
    r2 = x0 * x0 + x1 * x1
    s = 1.0 - r2
    omega = (s * s) * np.exp(0.25 * r2)
    out = np.zeros((x.shape[0], 2))
    out[:, 0] = -omega * x1
    out[:, 1] = omega * x0
    return out


def velocity_exact(t: float, x: np.ndarray) -> np.ndarray:
    return amplitude(t) * velocity_shape(x)


def pressure_exact(x: np.ndarray) -> np.ndarray:
    return 0.25 * np.sin(2.0 * x[:, 0] - x[:, 1]) + 0.1 * x[:, 0] * x[:, 1]


def euler_forcing(t: float, x: np.ndarray) -> np.ndarray:
    a = amplitude(t)
    out = amplitude_dt(t) * velocity_shape(x)
    v = velocity_shape(x)
    eps_val = 1.0e-6
    xpx = x.copy()
    xmx = x.copy()
    xpy = x.copy()
    xmy = x.copy()
    xpx[:, 0] += eps_val
    xmx[:, 0] -= eps_val
    xpy[:, 1] += eps_val
    xmy[:, 1] -= eps_val
    dvdx = (velocity_shape(xpx) - velocity_shape(xmx)) / (2.0 * eps_val)
    dvdy = (velocity_shape(xpy) - velocity_shape(xmy)) / (2.0 * eps_val)
    out[:, 0] += a * a * (v[:, 0] * dvdx[:, 0] + v[:, 1] * dvdy[:, 0])
    out[:, 1] += a * a * (v[:, 0] * dvdx[:, 1] + v[:, 1] * dvdy[:, 1])
    phase = 2.0 * x[:, 0] - x[:, 1]
    out[:, 0] += 0.5 * np.cos(phase) + 0.1 * x[:, 1]
    out[:, 1] += -0.25 * np.cos(phase) + 0.1 * x[:, 0]
    return out


def relative_l2(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u - v) / max(np.linalg.norm(v), 1.0e-14))


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


def test_pusl_incompressible_euler_manufactured_check():
    xi_u = 4
    xi_p = 4
    dual = build_dual_disk_domain(h=0.10, pressure_fraction=0.40, min_pressure_nodes=43)
    velocity_sp, pressure_sp = build_stencil_properties(dual.get_dim())

    solver = solvers.PUSLIncompressibleEulerSolver()
    dt = 0.02
    final_time = 0.04
    solver.init(dual, xi_u, velocity_sp, pressure_sp, dt)
    solver.set_tangential_flow_boundary(1.0e-5)

    xu = solver.get_output_nodes()
    problem = IncompressibleEulerBDFBackend.default_problem_definition()
    problem["slip_walls"] = [IncompressibleEulerBDFBackend.stationary_slip_wall(np.arange(1, solver.backend.xb.shape[0] + 1))]
    problem["gauge_options"]["mode"] = "forcepressuremean"

    u0 = velocity_exact(0.0, xu)
    solver.set_initial_velocity(u0)
    solver.bdf1_step(dt, None, euler_forcing, problem)
    sol = solver.bdf2_step(final_time, None, euler_forcing, problem)

    u_exact = velocity_exact(final_time, xu)
    xp = dual.get_pressure_domain().get_int_bdry_nodes()
    p_exact = pressure_exact(xp)
    p_num = sol["pressure"] - np.mean(sol["pressure"] - p_exact)

    u_rel = relative_l2(sol["velocity"], u_exact)
    p_rel = relative_l2(p_num, p_exact)

    assert np.isfinite(u_rel)
    assert np.isfinite(p_rel)
    assert u_rel < 3.0e-1
    assert p_rel < 6.0e-1
    assert sol["divergence_rms"] < 5.0e-1
    assert sol["divergence_max"] < 2.0
    assert sol["wall_normal_rms"] < 5.0e-2
    assert sol["wall_normal_max"] < 2.0e-1
