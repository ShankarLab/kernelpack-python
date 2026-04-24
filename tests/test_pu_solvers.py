import numpy as np

from kernelpack import geometry, nodes, solvers
from kernelpack.solvers._pu import pu_localized_evaluate, pu_patch_geometry


def build_test_domain():
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    curve = np.column_stack([np.cos(t), 0.8 * np.sin(t)])
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, 0.06, curve.shape[0])
    surface.build_level_set_from_geometric_model()
    generator = nodes.DomainNodeGenerator()
    return generator.build_domain_descriptor_from_geometry(
        surface,
        0.1,
        seed=17,
        strip_count=5,
        do_outer_refinement=True,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=2.0,
    )


def test_pu_localized_evaluate_reproduces_nodal_values():
    domain = build_test_domain()
    coeffs = np.sum(domain.get_all_nodes() ** 2, axis=1)
    patch_data = pu_patch_geometry(domain, 3)
    values = pu_localized_evaluate(domain, patch_data, 3, coeffs, domain.get_all_nodes()).reshape(-1)
    assert np.max(np.abs(values - coeffs)) < 5e-2


def test_pu_diffusion_solver_checks():
    domain = build_test_domain()
    nu = 0.25
    dt = 0.02
    xphys = domain.get_int_bdry_nodes()
    u_exact = lambda time, x: np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2)
    forcing = lambda nu_value, time, x: -np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2) - 4 * nu_value * np.exp(-time)
    neu_coeff_fixed = lambda xb: np.zeros(xb.shape[0])
    dir_coeff_fixed = lambda xb: np.ones(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, time, xb: u_exact(time, xb)

    solver = solvers.PUDiffusionSolver()
    solver.init(domain, 3, dt, nu)
    solver.set_initial_state(u_exact(0.0, xphys))
    u1 = solver.bdf1_step(dt, forcing, neu_coeff_fixed, dir_coeff_fixed, bc)
    assert np.max(np.abs(u1 - u_exact(dt, xphys))) < 4e-1

    solver2 = solvers.PUDiffusionSolver()
    solver2.init(domain, 3, dt, nu)
    solver2.set_state_history(u_exact(0.0, xphys), u_exact(dt, xphys))
    u2 = solver2.bdf2_step(2 * dt, forcing, neu_coeff_fixed, dir_coeff_fixed, bc)
    assert np.max(np.abs(u2 - u_exact(2 * dt, xphys))) < 5e-1


def test_multispecies_pu_diffusion_solver_checks():
    domain = build_test_domain()
    nu = 0.2
    dt = 0.02
    xphys = domain.get_int_bdry_nodes()
    u_exact = lambda time, x: np.column_stack(
        [
            np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2),
            np.exp(-2 * time) * (1 + x[:, 0]),
        ]
    )
    forcing = lambda nu_value, time, x: np.column_stack(
        [
            -np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2) - 4 * nu_value * np.exp(-time),
            -2 * np.exp(-2 * time) * (1 + x[:, 0]),
        ]
    )
    neu_coeff_fixed = lambda xb: np.zeros(xb.shape[0])
    dir_coeff_fixed = lambda xb: np.ones(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, time, xb: u_exact(time, xb)

    solver = solvers.MultiSpeciesPUDiffusionSolver()
    solver.init(domain, 3, dt, nu)
    solver.set_initial_state(u_exact(0.0, xphys))
    u1 = solver.bdf1_step(dt, forcing, neu_coeff_fixed, dir_coeff_fixed, bc)
    assert np.max(np.abs(u1 - u_exact(dt, xphys))) < 5e-1


def test_pusl_advection_constant_preservation():
    domain = build_test_domain()
    solver = solvers.PUSLAdvectionSolver()
    solver.init(domain, 4, 0.01)
    c0 = solver.project_constant(2.0, 1)
    velocity = lambda t, x: np.column_stack([-x[:, 1], x[:, 0]])
    c1 = solver.backward_sl_step(0.0, c0, velocity)
    assert np.max(np.abs(c1 - 2.0)) < 5.0e-4


def test_pusl_fd_advection_diffusion_smoke():
    domain = build_test_domain()
    solver = solvers.PUSLFDAdvectionDiffusionSolver()
    solver.init(domain, 4, 4, 0.01, 0.05, "backward")
    xout = solver.get_output_nodes()
    u0 = 1.0 + xout[:, 0]
    solver.set_initial_state(u0)
    velocity = lambda t, x: np.column_stack([-x[:, 1], x[:, 0]])
    forcing = lambda nu_value, t, x: np.zeros(x.shape[0])
    neu_coeff = lambda t, xb: np.ones(xb.shape[0])
    dir_coeff = lambda t, xb: np.zeros(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, t, xb: np.zeros(xb.shape[0])
    u1 = solver.bdf1_step(0.01, velocity, None, forcing, neu_coeff, dir_coeff, bc)
    assert u1.shape == u0.shape
    assert np.all(np.isfinite(u1))


def test_pusl_pu_advection_diffusion_smoke():
    domain = build_test_domain()
    solver = solvers.PUSLPUAdvectionDiffusionSolver()
    solver.init(domain, 4, 4, 0.01, 0.05, "backward")
    xout = solver.get_output_nodes()
    u0 = 1.0 + xout[:, 0]
    solver.set_initial_state(u0)
    velocity = lambda t, x: np.column_stack([-x[:, 1], x[:, 0]])
    forcing = lambda nu_value, t, x: np.zeros(x.shape[0])
    neu_coeff = lambda t, xb: np.ones(xb.shape[0])
    dir_coeff = lambda t, xb: np.zeros(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, t, xb: np.zeros(xb.shape[0])
    u1 = solver.bdf1_step(0.01, velocity, None, forcing, neu_coeff, dir_coeff, bc)
    assert u1.shape == u0.shape
    assert np.all(np.isfinite(u1))
