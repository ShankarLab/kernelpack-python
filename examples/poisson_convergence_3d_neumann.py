from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kernelpack import geometry, nodes, solvers


@dataclass
class ConvergenceRow:
    order: int
    h: float
    boundary_h: float
    n_physical: int
    linf: float
    l2: float


EXP_COEFFS = np.array([0.35, -0.2, 0.25], dtype=float)
SIN_COEFFS = np.array([1.1, -0.7, 0.9], dtype=float)
SIN_SCALE = 0.15


def exact_solution(x: np.ndarray) -> np.ndarray:
    phase_exp = x @ EXP_COEFFS
    phase_sin = x @ SIN_COEFFS
    return np.exp(phase_exp) + SIN_SCALE * np.sin(phase_sin)


def forcing_values(x: np.ndarray) -> np.ndarray:
    phase_exp = x @ EXP_COEFFS
    phase_sin = x @ SIN_COEFFS
    exp_term = -float(EXP_COEFFS @ EXP_COEFFS) * np.exp(phase_exp)
    sin_term = SIN_SCALE * float(SIN_COEFFS @ SIN_COEFFS) * np.sin(phase_sin)
    return exp_term + sin_term


def boundary_flux(xb: np.ndarray, nr: np.ndarray) -> np.ndarray:
    phase_exp = xb @ EXP_COEFFS
    phase_sin = xb @ SIN_COEFFS
    grad = np.exp(phase_exp)[:, None] * EXP_COEFFS[None, :] + SIN_SCALE * np.cos(phase_sin)[:, None] * SIN_COEFFS[None, :]
    return np.sum(grad * nr, axis=1)


def make_smooth_surface_sites(n: int) -> np.ndarray:
    x = geometry.fibonacci_sphere(n)
    uv = geometry.cart2sph_rows(x)
    r = 1.0 + 0.12 * np.cos(3.0 * uv[:, 0]) * np.cos(2.0 * uv[:, 1])
    return x * r[:, None]


def build_domain(surface_sites: np.ndarray, h: float, boundary_h: float) -> object:
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(surface_sites)
    surface.build_closed_geometric_model_ps(3, boundary_h, surface_sites.shape[0])
    surface.build_level_set_from_geometric_model()

    generator = nodes.DomainNodeGenerator()
    return generator.build_domain_descriptor_from_geometry(
        surface,
        h,
        seed=17,
        strip_count=6,
        do_outer_refinement=True,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=2.0,
    )


def estimate_rates(rows: list[list[ConvergenceRow]]) -> list[dict[str, object]]:
    rate_rows: list[dict[str, object]] = []
    for order_rows in rows:
        h = np.asarray([row.h for row in order_rows], dtype=float)
        linf = np.asarray([row.linf for row in order_rows], dtype=float)
        l2 = np.asarray([row.l2 for row in order_rows], dtype=float)
        rate_rows.append(
            {
                "order": order_rows[0].order,
                "linf": np.log(linf[:-1] / linf[1:]) / np.log(h[:-1] / h[1:]),
                "l2": np.log(l2[:-1] / l2[1:]) / np.log(h[:-1] / h[1:]),
            }
        )
    return rate_rows


def print_results(rows: list[list[ConvergenceRow]], rates: list[dict[str, object]], backend: str, assembler: str) -> None:
    print()
    print(f"3D Poisson pure-Neumann convergence study ({backend.upper()}, {assembler.upper()})")
    print("Exact solution: exp(0.35x - 0.2y + 0.25z) + 0.15 sin(1.1x - 0.7y + 0.9z)")
    print()
    for order_rows, rate_row in zip(rows, rates, strict=True):
        print(f"Order {order_rows[0].order}")
        print("  h        hb       Nphys      Linf error      L2 error        Linf rate   L2 rate")
        for ih, row in enumerate(order_rows):
            if ih == 0:
                print(
                    f"  {row.h:<7.3f}  {row.boundary_h:<7.3f}  {row.n_physical:<9d}  "
                    f"{row.linf:<14.6e}  {row.l2:<14.6e}  {'-':<10}  {'-':<10}"
                )
            else:
                print(
                    f"  {row.h:<7.3f}  {row.boundary_h:<7.3f}  {row.n_physical:<9d}  "
                    f"{row.linf:<14.6e}  {row.l2:<14.6e}  "
                    f"{rate_row['linf'][ih - 1]:<10.4f}  {rate_row['l2'][ih - 1]:<10.4f}"
                )
        print()


def plot_results(rows: list[list[ConvergenceRow]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    markers = {2: "o", 4: "s", 6: "^"}
    colors = {2: "tab:blue", 4: "tab:orange", 6: "tab:green"}
    for order_rows in rows:
        order = order_rows[0].order
        h = np.asarray([row.h for row in order_rows], dtype=float)
        linf = np.asarray([row.linf for row in order_rows], dtype=float)
        l2 = np.asarray([row.l2 for row in order_rows], dtype=float)
        axes[0].loglog(h, linf, marker=markers[order], color=colors[order], linewidth=1.8, label=f"order {order}")
        axes[1].loglog(h, l2, marker=markers[order], color=colors[order], linewidth=1.8, label=f"order {order}")
    axes[0].set_title(r"$L^\infty$ Error")
    axes[1].set_title(r"$L^2$ Error")
    for ax in axes:
        ax.set_xlabel("h")
        ax.set_ylabel("error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        ax.invert_xaxis()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_convergence_study(
    backend: str = "rbf",
    *,
    assembler: str = "fd",
    orders: tuple[int, ...] = (2, 4, 6),
    h_values: tuple[float, ...] = (0.28, 0.22, 0.18),
    boundary_h_values: tuple[float, ...] | None = None,
    surface_site_count: int = 220,
) -> dict[str, object]:
    surface_sites = make_smooth_surface_sites(surface_site_count)
    if boundary_h_values is None:
        boundary_h_values = h_values
    if len(boundary_h_values) != len(h_values):
        raise ValueError("boundary_h_values must match h_values in length")

    rows: list[list[ConvergenceRow]] = []
    neu_coeff = lambda xb: np.ones(xb.shape[0])
    dir_coeff = lambda xb: np.zeros(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, xb: boundary_flux(xb, nr)

    for order in orders:
        order_rows: list[ConvergenceRow] = []
        for h, boundary_h in zip(h_values, boundary_h_values, strict=True):
            domain = build_domain(surface_sites, h, boundary_h)
            solver = solvers.PoissonSolver(
                lap_assembler=assembler,
                bc_assembler=assembler,
                lap_stencil=backend,
                bc_stencil=backend,
            )
            solver.init(domain, order)
            solve_result = solver.solve(forcing_values, neu_coeff, dir_coeff, bc)

            xphys = domain.get_int_bdry_nodes()
            u_true = exact_solution(xphys)
            err = solve_result["u"] - u_true
            err = err - np.mean(err)
            order_rows.append(
                ConvergenceRow(
                    order=order,
                    h=h,
                    boundary_h=boundary_h,
                    n_physical=xphys.shape[0],
                    linf=float(np.max(np.abs(err))),
                    l2=float(np.linalg.norm(err) / np.sqrt(err.size)),
                )
            )
        rows.append(order_rows)

    rates = estimate_rates(rows)
    return {
        "orders": list(orders),
        "h": list(h_values),
        "boundary_h": list(boundary_h_values),
        "backend": backend,
        "assembler": assembler,
        "rows": [[asdict(row) for row in order_rows] for order_rows in rows],
        "rates": [
            {
                "order": int(rate_row["order"]),
                "linf": np.asarray(rate_row["linf"], dtype=float).tolist(),
                "l2": np.asarray(rate_row["l2"], dtype=float).tolist(),
            }
            for rate_row in rates
        ],
        "raw_rows": rows,
        "raw_rates": rates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 3D pure-Neumann Poisson convergence study for KernelPack Python.")
    parser.add_argument("--backend", default="rbf", choices=["rbf", "wls"], help="Stencil backend to use.")
    parser.add_argument("--assembler", default="fd", choices=["fd", "fdo"], help="Assembler to use.")
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 4, 6], help="Target convergence orders.")
    parser.add_argument("--h-values", nargs="+", type=float, default=[0.28, 0.22, 0.18], help="Interior node spacings.")
    parser.add_argument("--boundary-h-values", nargs="+", type=float, default=None, help="Boundary node spacings.")
    parser.add_argument("--surface-site-count", type=int, default=220, help="Number of surface data sites used to define the geometry.")
    parser.add_argument("--output-dir", default="artifacts/convergence", help="Directory for JSON and figure outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    boundary_h_values = tuple(args.boundary_h_values) if args.boundary_h_values is not None else None
    results = run_convergence_study(
        args.backend,
        assembler=args.assembler,
        orders=tuple(args.orders),
        h_values=tuple(args.h_values),
        boundary_h_values=boundary_h_values,
        surface_site_count=args.surface_site_count,
    )
    rows = results.pop("raw_rows")
    rates = results.pop("raw_rates")
    print_results(rows, rates, results["backend"], results["assembler"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"poisson_convergence_3d_neumann_{results['backend']}_{results['assembler']}"
    json_path = output_dir / f"{stem}.json"
    fig_path = output_dir / f"{stem}.png"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_results(rows, fig_path)
    print(f"Saved results to {json_path}")
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
