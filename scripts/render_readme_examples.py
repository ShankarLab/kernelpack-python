from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from kernelpack.geometry import EmbeddedSurface
from kernelpack.nodes import DomainNodeGenerator
from kernelpack.solvers import DiffusionSolver, PoissonSolver


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "docs" / "readme_assets"
OUTDIR.mkdir(parents=True, exist_ok=True)


def build_domain(*, do_outer_refinement: bool = True) -> tuple[EmbeddedSurface, object]:
    t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    curve = np.column_stack([np.cos(t), 0.7 * np.sin(t)])

    surface = EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, 0.06, curve.shape[0])
    surface.build_level_set_from_geometric_model()

    generator = DomainNodeGenerator()
    domain = generator.build_domain_descriptor_from_geometry(
        surface,
        0.08,
        seed=17,
        strip_count=5,
        do_outer_refinement=do_outer_refinement,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=2.0,
    )
    return surface, domain


def save_geometry(surface, domain) -> Path:
    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    axes = fig.subplot_mosaic([["sites", "boundary"], ["domain", "domain"]])
    xb = domain.get_bdry_nodes()
    nr = domain.get_nrmls()
    xi = domain.get_interior_nodes()
    xg = domain.get_ghost_nodes()

    axes["sites"].plot(surface.data_sites[:, 0], surface.data_sites[:, 1], "ko", ms=3)
    axes["sites"].set_title("Input Sites")

    axes["boundary"].plot(xb[:, 0], xb[:, 1], ".", color="#0f766e", ms=4)
    step = max(1, xb.shape[0] // 40)
    axes["boundary"].quiver(
        xb[::step, 0],
        xb[::step, 1],
        nr[::step, 0],
        nr[::step, 1],
        angles="xy",
        scale_units="xy",
        scale=18,
        color="#b91c1c",
        width=0.003,
    )
    axes["boundary"].set_title("Boundary Samples and Normals")

    axes["domain"].plot(xi[:, 0], xi[:, 1], ".", color="#1d4ed8", ms=3, label="Interior")
    axes["domain"].plot(xb[:, 0], xb[:, 1], ".", color="#0f766e", ms=3, label="Boundary")
    axes["domain"].plot(xg[:, 0], xg[:, 1], ".", color="#f59e0b", ms=3, label="Ghost")
    axes["domain"].legend(frameon=False, fontsize=9)
    axes["domain"].set_title("Boundary-Refined Interior, Boundary, and Ghost Nodes")

    for ax in axes.values():
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    path = OUTDIR / "geometry_domain.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_poisson(domain) -> Path:
    solver = PoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="rbf",
        bc_stencil="rbf",
    )
    solver.init(domain, 4)

    u_exact = lambda x: (x[:, 0] ** 2 + x[:, 1] ** 2) ** 2 - (x[:, 0] ** 2 + x[:, 1] ** 2) + 1.0 / 6.0
    forcing = lambda x: 4.0 - 16.0 * (x[:, 0] ** 2 + x[:, 1] ** 2)
    neu_coeff = lambda xb: np.ones(xb.shape[0])
    dir_coeff = lambda xb: np.zeros(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, xb: np.sum(
        np.column_stack(
            [
                4.0 * xb[:, 0] * (xb[:, 0] ** 2 + xb[:, 1] ** 2) - 2.0 * xb[:, 0],
                4.0 * xb[:, 1] * (xb[:, 0] ** 2 + xb[:, 1] ** 2) - 2.0 * xb[:, 1],
            ]
        )
        * nr,
        axis=1,
    )

    result = solver.solve(forcing, neu_coeff, dir_coeff, bc)
    x_phys = domain.get_int_bdry_nodes()
    u = result["u"]
    u_true = u_exact(x_phys)
    u = u - np.mean(u - u_true)
    err = u - u_true

    tri = mtri.Triangulation(x_phys[:, 0], x_phys[:, 1])

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    cf0 = axes[0].tricontourf(tri, u, levels=24, cmap="viridis")
    axes[0].plot(domain.get_bdry_nodes()[:, 0], domain.get_bdry_nodes()[:, 1], "k.", ms=1.5, alpha=0.5)
    axes[0].set_title("Poisson Solution")
    fig.colorbar(cf0, ax=axes[0], shrink=0.9)

    cf1 = axes[1].tricontourf(tri, err, levels=24, cmap="coolwarm")
    axes[1].set_title(f"Poisson Error\nmax |e| = {np.max(np.abs(err)):.2e}")
    fig.colorbar(cf1, ax=axes[1], shrink=0.9)

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)

    path = OUTDIR / "poisson_solution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_diffusion(domain) -> Path:
    solver = DiffusionSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="rbf",
        bc_stencil="rbf",
    )
    nu = 0.25
    dt = 0.02
    t_final = 0.50
    nsteps = int(round(t_final / dt))
    solver.init(domain, 4, dt, nu)

    x_phys = domain.get_int_bdry_nodes()
    u_exact = lambda time, x: np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2)
    forcing = lambda nu_value, time, x: -np.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2) - 4.0 * nu_value * np.exp(-time)
    neu_coeff = lambda xb: np.zeros(xb.shape[0])
    dir_coeff = lambda xb: np.ones(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, time, xb: u_exact(time, xb)

    solver.set_initial_state(u_exact(0.0, x_phys))
    times = [0.0]
    errors = [0.0]

    for step in range(1, nsteps + 1):
        time = step * dt
        if step == 1:
            u_next = solver.bdf1_step(time, forcing, neu_coeff, dir_coeff, bc)
        elif step == 2:
            u_next = solver.bdf2_step(time, forcing, neu_coeff, dir_coeff, bc)
        else:
            u_next = solver.bdf3_step(time, forcing, neu_coeff, dir_coeff, bc)
        times.append(time)
        errors.append(float(np.max(np.abs(u_next - u_exact(time, x_phys)))))

    u_final = solver.current_physical_state()
    u_true_final = u_exact(t_final, x_phys)
    err = u_final - u_true_final
    tri = mtri.Triangulation(x_phys[:, 0], x_phys[:, 1])

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    axes = fig.subplot_mosaic([["solution", "error"], ["history", "history"]])
    cf0 = axes["solution"].tricontourf(tri, u_final, levels=24, cmap="viridis")
    axes["solution"].set_title(f"Diffusion at t = {t_final:.2f}")
    fig.colorbar(cf0, ax=axes["solution"], shrink=0.85)

    cf1 = axes["error"].tricontourf(tri, err, levels=24, cmap="coolwarm")
    axes["error"].set_title(f"Final-Time Error\nmax |e| = {np.max(np.abs(err)):.2e}")
    fig.colorbar(cf1, ax=axes["error"], shrink=0.85)

    axes["history"].plot(times, errors, color="#1d4ed8", lw=2)
    axes["history"].scatter(times, errors, color="#0f766e", s=18)
    axes["history"].set_title("Time March to Final Time")
    axes["history"].set_xlabel("t")
    axes["history"].set_ylabel("max nodal error")
    axes["history"].grid(alpha=0.2)

    for ax in (axes["solution"], axes["error"]):
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)

    path = OUTDIR / "diffusion_solution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    surface, geometry_domain = build_domain(do_outer_refinement=True)
    _, solver_domain = build_domain(do_outer_refinement=True)
    paths = [
        save_geometry(surface, geometry_domain),
        save_poisson(solver_domain),
        save_diffusion(solver_domain),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
