import numpy as np

from kernelpack import geometry, nodes, solvers


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


def test_poisson_solver_checks():
    domain = build_test_domain()
    u_exact = lambda x: x[:, 0] ** 2 + x[:, 1] ** 2
    u_true = u_exact(domain.get_int_bdry_nodes())

    wls_solver = solvers.PoissonSolver(lap_assembler="fd", bc_assembler="fd", lap_stencil="wls", bc_stencil="wls")
    wls_solver.init(domain, 3)
    wls_result = wls_solver.solve(
        lambda xeq: -4 * np.ones(xeq.shape[0]),
        lambda xb: np.zeros(xb.shape[0]),
        lambda xb: np.ones(xb.shape[0]),
        lambda neu_coeffs, dir_coeffs, nr, xb: u_exact(xb),
    )
    assert np.max(np.abs(wls_result["u"] - u_true)) < 2.5e-1

    rbf_solver = solvers.PoissonSolver(lap_assembler="fd", bc_assembler="fd", lap_stencil="rbf", bc_stencil="rbf")
    rbf_solver.init(domain, 3)
    rbf_result = rbf_solver.solve(
        lambda xeq: -4 * np.ones(xeq.shape[0]),
        lambda xb: np.zeros(xb.shape[0]),
        lambda xb: np.ones(xb.shape[0]),
        lambda neu_coeffs, dir_coeffs, nr, xb: u_exact(xb),
    )
    assert np.max(np.abs(rbf_result["u"] - u_true)) < 6e-1

    zero_neu_solver = solvers.PoissonSolver(lap_assembler="fd", bc_assembler="fd", lap_stencil="wls", bc_stencil="wls")
    zero_neu_solver.init(domain, 3)
    zero_neu_result = zero_neu_solver.solve(
        lambda xeq: np.zeros(xeq.shape[0]),
        lambda xb: np.ones(xb.shape[0]),
        lambda xb: np.zeros(xb.shape[0]),
        lambda neu_coeffs, dir_coeffs, nr, xb: np.zeros(xb.shape[0]),
    )
    assert zero_neu_result["used_nullspace_augmentation"]
    assert np.max(np.abs(zero_neu_result["u"])) < 1e-8
    assert zero_neu_solver.last_solve_used_nullspace()


def test_variable_poisson_solver_checks():
    domain = build_test_domain()
    u_exact = lambda x: x[:, 0] ** 2 + x[:, 1] ** 2
    a_coeff = lambda x: 2 + x[:, 0] + 0.2 * x[:, 1]
    forcing = lambda x: -(4 * a_coeff(x) + 2 * x[:, 0] + 0.4 * x[:, 1])
    u_true = u_exact(domain.get_int_bdry_nodes())

    wls_solver = solvers.VariablePoissonSolver(lap_assembler="fd", bc_assembler="fd", lap_stencil="wls", bc_stencil="wls")
    wls_solver.init(domain, 3)
    wls_result = wls_solver.solve(
        forcing,
        a_coeff,
        lambda xb: np.zeros(xb.shape[0]),
        lambda xb: np.ones(xb.shape[0]),
        lambda neu_coeffs, dir_coeffs, nr, xb: u_exact(xb),
    )
    assert np.max(np.abs(wls_result["u"] - u_true)) < 4e-1

    rbf_solver = solvers.VariablePoissonSolver(lap_assembler="fd", bc_assembler="fd", lap_stencil="rbf", bc_stencil="rbf")
    rbf_solver.init(domain, 3)
    rbf_result = rbf_solver.solve(
        forcing,
        a_coeff,
        lambda xb: np.zeros(xb.shape[0]),
        lambda xb: np.ones(xb.shape[0]),
        lambda neu_coeffs, dir_coeffs, nr, xb: u_exact(xb),
    )
    assert np.max(np.abs(rbf_result["u"] - u_true)) < 6e-1
    assert rbf_result["PDE"].shape[0] == domain.get_num_int_bdry_nodes()

    zero_neu_solver = solvers.VariablePoissonSolver(lap_assembler="fd", bc_assembler="fd", lap_stencil="wls", bc_stencil="wls")
    zero_neu_solver.init(domain, 3)
    zero_neu_result = zero_neu_solver.solve(
        lambda xeq: np.zeros(xeq.shape[0]),
        lambda xeq: np.ones(xeq.shape[0]),
        lambda xb: np.ones(xb.shape[0]),
        lambda xb: np.zeros(xb.shape[0]),
        lambda neu_coeffs, dir_coeffs, nr, xb: np.zeros(xb.shape[0]),
    )
    assert zero_neu_result["used_nullspace_augmentation"]
    assert np.max(np.abs(zero_neu_result["u"])) < 1e-8


def test_diffusion_solver_checks():
    domain = build_test_domain()
    nu = 0.25
    dt = 0.02
    xphys = domain.get_int_bdry_nodes()
    u_exact = lambda time, x: np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2)
    forcing = lambda nu_value, time, x: -np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2) - 4 * nu_value * np.exp(-time)
    neu_coeff_fixed = lambda xb: np.zeros(xb.shape[0])
    dir_coeff_fixed = lambda xb: np.ones(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, time, xb: u_exact(time, xb)

    wls_solver = solvers.DiffusionSolver(lap_assembler="fd", bc_assembler="fd", lap_stencil="wls", bc_stencil="wls")
    wls_solver.init(domain, 3, dt, nu)
    wls_solver.set_initial_state(u_exact(0, xphys))
    u1 = wls_solver.bdf1_step(dt, forcing, neu_coeff_fixed, dir_coeff_fixed, bc)
    assert np.max(np.abs(u1 - u_exact(dt, xphys))) < 3e-1
    u2 = wls_solver.bdf2_step(2 * dt, forcing, neu_coeff_fixed, dir_coeff_fixed, bc)
    assert np.max(np.abs(u2 - u_exact(2 * dt, xphys))) < 3.5e-1
    u3 = wls_solver.bdf3_step(3 * dt, forcing, neu_coeff_fixed, dir_coeff_fixed, bc)
    assert np.max(np.abs(u3 - u_exact(3 * dt, xphys))) < 4e-1

    rbf_solver = solvers.DiffusionSolver(lap_assembler="fd", bc_assembler="fd", lap_stencil="rbf", bc_stencil="rbf")
    rbf_solver.init(domain, 3, dt, nu)
    rbf_solver.set_initial_state(u_exact(0, xphys))
    u1_rbf = rbf_solver.bdf1_step(
        dt,
        forcing,
        lambda time, xb: np.zeros(xb.shape[0]),
        lambda time, xb: np.ones(xb.shape[0]),
        bc,
    )
    assert np.max(np.abs(u1_rbf - u_exact(dt, xphys))) < 7e-1

    history_solver = solvers.DiffusionSolver(lap_assembler="fd", bc_assembler="fd", lap_stencil="wls", bc_stencil="wls")
    history_solver.init(domain, 3, dt, nu)
    history_solver.set_state_history(u_exact(0, xphys), u_exact(dt, xphys), u_exact(2 * dt, xphys))
    state_now = history_solver.current_physical_state()
    assert np.max(np.abs(state_now - u_exact(2 * dt, xphys))) < 1e-12
