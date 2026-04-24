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


def test_heterogeneous_multispecies_pu_diffusion_solver_checks():
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

    solver = solvers.HeterogeneousMultiSpeciesPUDiffusionSolver()
    solver.init(domain, 3, dt, nus)
    u0 = np.column_stack([u_exact(species, 0.0, xphys) for species in range(nus.size)])
    solver.set_initial_state(u0)
    u1 = solver.bdf1_step(dt, forcing, neu_coeff, dir_coeff, bc)
    target = np.column_stack([u_exact(species, dt, xphys) for species in range(nus.size)])
    assert np.max(np.abs(u1 - target)) < 6e-1


def test_pusl_advection_constant_preservation():
    domain = build_test_domain()
    solver = solvers.PUSLAdvectionSolver()
    solver.init(domain, 4, 0.01)
    c0 = solver.project_constant(2.0, 1)
    velocity = lambda t, x: np.column_stack([-x[:, 1], x[:, 0]])
    c1 = solver.backward_sl_step(0.0, c0, velocity)
    assert np.max(np.abs(c1 - 2.0)) < 5.0e-4


def test_pusl_forward_advection_uses_defect_correction():
    domain = build_test_domain()
    solver = solvers.PUSLAdvectionSolver()
    solver.init(domain, 4, 0.01)
    xout = solver.get_output_nodes()
    c0 = (1.0 + 0.25 * xout[:, 0] - 0.1 * xout[:, 1]).reshape(-1, 1)
    velocity = lambda t, x: np.column_stack([-x[:, 1], x[:, 0]])
    solver.reset_solve_stats()
    c1 = solver.forward_sl_step(0.0, c0, velocity)
    stats = solver.get_solve_stats()
    assert c1.shape == c0.shape
    assert np.all(np.isfinite(c1))
    assert stats["defect_correction_solves"] >= 1


def test_pusl_forward_inflow_falls_back_to_backward():
    domain = build_test_domain()
    solver = solvers.PUSLAdvectionSolver()
    solver.init(domain, 4, 0.01)
    solver.set_inflow_dirichlet_boundary(lambda t, x: np.ones((x.shape[0], 1)))
    xout = solver.get_output_nodes()
    c0 = (1.0 + xout[:, 0]).reshape(-1, 1)
    velocity = lambda t, x: 2.0 * x
    forward = solver.forward_sl_step(0.0, c0, velocity)
    backward = solver.backward_sl_step(0.0, c0, velocity)
    assert np.max(np.abs(forward - backward)) < 1.0e-12


def test_pusl_tangential_boundary_validation_raises_for_normal_flow():
    domain = build_test_domain()
    solver = solvers.PUSLAdvectionSolver()
    solver.init(domain, 4, 0.01)
    solver.set_tangential_flow_boundary()
    c0 = solver.project_constant(1.0, 1)
    velocity = lambda t, x: x
    try:
        solver.backward_sl_step(0.0, c0, velocity)
    except ValueError as exc:
        assert "tangential-flow" in str(exc)
    else:
        raise AssertionError("expected tangential-flow validation to fail for normal velocity")


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


def test_pusl_fd_advection_diffusion_reaction_changes_solution():
    domain = build_test_domain()
    dt = 0.01
    nu = 0.05
    velocity = lambda t, x: np.column_stack([-x[:, 1], x[:, 0]])
    forcing = lambda nu_value, t, x: np.zeros(x.shape[0])
    reaction = lambda t, state, x: 0.5 * np.ones(x.shape[0])
    neu_coeff = lambda t, xb: np.ones(xb.shape[0])
    dir_coeff = lambda t, xb: np.zeros(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, t, xb: np.zeros(xb.shape[0])

    solver_no_rxn = solvers.PUSLFDAdvectionDiffusionSolver()
    solver_no_rxn.init(domain, 4, 4, dt, nu, "backward")
    xout = solver_no_rxn.get_output_nodes()
    u0 = 1.0 + xout[:, 0]
    solver_no_rxn.set_initial_state(u0)
    u1_no = solver_no_rxn.bdf1_step(dt, velocity, None, forcing, neu_coeff, dir_coeff, bc)

    solver_rxn = solvers.PUSLFDAdvectionDiffusionSolver()
    solver_rxn.init(domain, 4, 4, dt, nu, "backward")
    solver_rxn.set_initial_state(u0)
    u1_yes = solver_rxn.bdf1_step(dt, velocity, None, forcing, neu_coeff, dir_coeff, bc, reaction=reaction)

    assert u1_yes.shape == u1_no.shape
    assert np.all(np.isfinite(u1_yes))
    assert np.max(np.abs(u1_yes - u1_no)) > 1.0e-8


def test_pusl_pu_advection_diffusion_reaction_wrapper_smoke():
    domain = build_test_domain()
    dt = 0.01
    nu = 0.05
    velocity = lambda t, x: np.column_stack([-x[:, 1], x[:, 0]])
    forcing = lambda nu_value, t, x: np.zeros(x.shape[0])
    reaction = lambda t, state, x: -0.1 * state
    neu_coeff = lambda t, xb: np.ones(xb.shape[0])
    dir_coeff = lambda t, xb: np.zeros(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, t, xb: np.zeros(xb.shape[0])

    solver = solvers.PUSLPUAdvectionDiffusionReactionSolver()
    solver.init(domain, 4, 4, dt, nu, "backward")
    xout = solver.get_output_nodes()
    u0 = 1.0 + xout[:, 0]
    solver.set_initial_state(u0)
    u1 = solver.bdf1_step(dt, velocity, None, forcing, reaction, neu_coeff, dir_coeff, bc)
    assert u1.shape == u0.shape
    assert np.all(np.isfinite(u1))
