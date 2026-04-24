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


def test_variable_poisson_solver_matches_mixed_bc_case():
    domain = build_test_domain()

    def exact_solution(x):
        return 1.0 + np.sin(x[:, 0]) * np.sin(x[:, 1]) + 0.25 * np.cos(0.5 * x[:, 0])

    def coeff_field(x):
        return 1.4 + 0.2 * np.sin(x[:, 0]) + 0.15 * np.cos(x[:, 1])

    def forcing(x):
        sx = np.sin(x[:, 0])
        sy = np.sin(x[:, 1])
        cx = np.cos(x[:, 0])
        cy = np.cos(x[:, 1])
        ux = cx * sy - 0.125 * np.sin(0.5 * x[:, 0])
        uy = sx * cy
        lap = -2.0 * sx * sy - 0.0625 * np.cos(0.5 * x[:, 0])
        ax = 0.2 * np.cos(x[:, 0])
        ay = -0.15 * np.sin(x[:, 1])
        return -coeff_field(x) * lap - ax * ux - ay * uy

    def normal_derivative(nr, x):
        dx = np.cos(x[:, 0]) * np.sin(x[:, 1]) - 0.125 * np.sin(0.5 * x[:, 0])
        dy = np.sin(x[:, 0]) * np.cos(x[:, 1])
        return nr[:, 0] * dx + nr[:, 1] * dy

    neu_coeff = lambda xb: 0.6 + 0.15 * xb[:, 0] ** 2 + 0.05 * xb[:, 1] ** 2
    dir_coeff = lambda xb: 1.0 + 0.1 * xb[:, 0] ** 2 + 0.08 * xb[:, 1] ** 2

    solver = solvers.VariablePoissonSolver(lap_assembler="fdo", bc_assembler="fd", lap_stencil="rbf", bc_stencil="rbf")
    solver.init(domain, 4)
    result = solver.solve(
        forcing,
        coeff_field,
        neu_coeff,
        dir_coeff,
        lambda neu, dir_, nr, xb: neu * normal_derivative(nr, xb) + dir_ * exact_solution(xb),
    )
    assert np.max(np.abs(result["u"] - exact_solution(domain.get_int_bdry_nodes()))) < 2.5e-1
    assert np.array_equal(solver.get_output_nodes(), domain.get_int_bdry_nodes())
    assert solver.get_output_range() == (0, domain.get_num_int_bdry_nodes())
    assert not solver.returns_distributed_state()


def test_nonlinear_variable_poisson_solver_checks():
    domain = build_test_domain()

    def exact_solution(x):
        return 1.0 + np.sin(x[:, 0]) * np.sin(x[:, 1]) + 0.25 * np.cos(0.5 * x[:, 0])

    def exp_scaled(u, scale):
        return np.exp(scale * u)

    def forcing(x, _u):
        sx = np.sin(x[:, 0])
        sy = np.sin(x[:, 1])
        cx = np.cos(x[:, 0])
        cy = np.cos(x[:, 1])
        ux = cx * sy - 0.125 * np.sin(0.5 * x[:, 0])
        uy = sx * cy
        lap = -2.0 * sx * sy - 0.0625 * np.cos(0.5 * x[:, 0])
        grad_sq = ux**2 + uy**2
        a = exp_scaled(exact_solution(x), 0.2)
        return -a * lap - 0.2 * a * grad_sq

    def forcing_u(x, u):
        return np.zeros(x.shape[0])

    def coeff(x, u):
        return exp_scaled(u, 0.2)

    def coeff_u(x, u):
        return 0.2 * exp_scaled(u, 0.2)

    def normal_derivative(nr, x):
        dx = np.cos(x[:, 0]) * np.sin(x[:, 1]) - 0.125 * np.sin(0.5 * x[:, 0])
        dy = np.sin(x[:, 0]) * np.cos(x[:, 1])
        return nr[:, 0] * dx + nr[:, 1] * dy

    neu_coeff = lambda xb: 0.6 + 0.15 * xb[:, 0] ** 2 + 0.05 * xb[:, 1] ** 2
    dir_coeff = lambda xb: 1.0 + 0.1 * xb[:, 0] ** 2 + 0.08 * xb[:, 1] ** 2

    solver = solvers.NonlinearVariablePoissonSolver(lap_assembler="fdo", bc_assembler="fd", lap_stencil="rbf", bc_stencil="rbf")
    solver.init(domain, 4)
    solver.set_nonlinear_tolerance(1.0e-10)
    solver.set_linear_tolerance(1.0e-10)
    solver.set_max_nonlinear_iterations(12)
    result = solver.solve(
        forcing,
        forcing_u,
        coeff,
        coeff_u,
        neu_coeff,
        dir_coeff,
        lambda neu, dir_, nr, xb: neu * normal_derivative(nr, xb) + dir_ * exact_solution(xb),
        initial_guess=exact_solution(domain.get_int_bdry_nodes()),
    )

    error = np.max(np.abs(result["u"] - exact_solution(domain.get_int_bdry_nodes())))
    assert error < 3.0e-1
    assert result["iterations"] <= 12
    assert result["residual_norm"] < 1.0e-8
    assert solver.get_last_nonlinear_iterations() == result["iterations"]
    assert solver.get_last_residual_norm() == result["residual_norm"]
    assert np.array_equal(solver.get_output_nodes(), domain.get_int_bdry_nodes())


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


def test_multispecies_diffusion_solver_checks():
    domain = build_test_domain()
    nu = 0.2
    dt = 0.02
    xphys = domain.get_int_bdry_nodes()
    u_exact = lambda time, x: np.column_stack(
        [
            np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2),
            2.0 * np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2),
            -0.5 * np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2),
        ]
    )
    forcing = lambda nu_value, time, x: np.column_stack(
        [
            -np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2) - 4 * nu_value * np.exp(-time),
            -2.0 * np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2) - 8 * nu_value * np.exp(-time),
            0.5 * np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2) + 2 * nu_value * np.exp(-time),
        ]
    )
    neu_coeff_fixed = lambda xb: np.zeros(xb.shape[0])
    dir_coeff_fixed = lambda xb: np.ones(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, time, xb: u_exact(time, xb)

    solver = solvers.MultiSpeciesDiffusionSolver(lap_assembler="fd", bc_assembler="fd", lap_stencil="wls", bc_stencil="wls")
    solver.init(domain, 3, dt, nu)
    solver.set_initial_state(u_exact(0.0, xphys))
    u1 = solver.bdf1_step(dt, forcing, neu_coeff_fixed, dir_coeff_fixed, bc)
    assert np.max(np.abs(u1 - u_exact(dt, xphys))) < 4e-1


def test_heterogeneous_multispecies_diffusion_solver_checks():
    domain = build_test_domain()
    nus = np.array([0.15, 0.25])
    dt = 0.02
    xphys = domain.get_int_bdry_nodes()

    def u_exact(species: int, time: float, x: np.ndarray) -> np.ndarray:
        amp = 1.0 + 0.5 * species
        return amp * np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2)

    def forcing(species: int, nu_value: float, time: float, x: np.ndarray) -> np.ndarray:
        amp = 1.0 + 0.5 * species
        return -amp * np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2) - 4 * amp * nu_value * np.exp(-time)

    neu_coeff = lambda species, time, xb: np.zeros(xb.shape[0])
    dir_coeff = lambda species, time, xb: np.ones(xb.shape[0])
    bc = lambda species, neu_coeffs, dir_coeffs, nr, time, xb: u_exact(species, time, xb)

    solver = solvers.HeterogeneousMultiSpeciesDiffusionSolver(lap_assembler="fd", bc_assembler="fd", lap_stencil="wls", bc_stencil="wls")
    solver.init(domain, 3, dt, nus)
    u0 = np.column_stack([u_exact(species, 0.0, xphys) for species in range(nus.size)])
    solver.set_initial_state(u0)
    u1 = solver.bdf1_step(dt, forcing, neu_coeff, dir_coeff, bc)
    target = np.column_stack([u_exact(species, dt, xphys) for species in range(nus.size)])
    assert np.max(np.abs(u1 - target)) < 4.5e-1
