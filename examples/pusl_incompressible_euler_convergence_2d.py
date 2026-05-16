from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kernelpack import geometry, nodes, solvers
from kernelpack.rbffd import StencilProperties
from kernelpack.solvers.detail import IncompressibleEulerBDFBackend


@dataclass
class CaseResult:
    xi_u: int
    xi_p: int
    h: float
    nu: int
    np_: int
    u_rel: float
    p_rel: float
    div_rms: float
    div_max: float
    wall_rms: float
    wall_max: float


def make_stencil_properties(xi: int) -> StencilProperties:
    sp = StencilProperties()
    sp.dim = 2
    theta = 2
    sp.ell = max(xi + theta - 1, 2)
    sp.npoly = int((sp.ell + 1) * (sp.ell + 2) / 2)
    sp.n = 2 * sp.npoly + 1
    sp.spline_degree = sp.ell
    if sp.spline_degree % 2 == 0:
        sp.spline_degree -= 1
    sp.spline_degree = max(sp.spline_degree, 5)
    sp.tree_mode = "interior_boundary"
    sp.point_set = "interior_boundary"
    return sp


def make_domain(h: float, pressure_fraction: float, min_pressure_nodes: int):
    t = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    boundary = np.column_stack([np.cos(t), np.sin(t)])
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(boundary)
    surface.build_closed_geometric_model_ps(2, h, boundary.shape[0])
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
    dual = generator.create_dual_node_domain_descriptor()
    dual.build_structs()
    return dual


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


def run_case(xi_u: int, xi_p: int, h: float, dt: float, final_time: float) -> tuple[CaseResult, dict[str, np.ndarray]]:
    velocity_sp = make_stencil_properties(xi_u)
    pressure_sp = make_stencil_properties(xi_p)
    dual = make_domain(h, 0.40, pressure_sp.n)

    solver = solvers.PUSLIncompressibleEulerSolver()
    solver.init(dual, xi_u, velocity_sp, pressure_sp, dt, 1)
    solver.set_tangential_flow_boundary(1.0e-5)

    xu = dual.get_velocity_domain().get_int_bdry_nodes()
    problem = IncompressibleEulerBDFBackend.default_problem_definition()
    problem["slip_walls"] = [IncompressibleEulerBDFBackend.stationary_slip_wall(np.arange(1, dual.get_velocity_domain().get_num_bdry_nodes() + 1))]
    problem["gauge_options"]["mode"] = "forcepressuremean"

    u0 = velocity_exact(0.0, xu)
    solver.set_initial_velocity(u0)
    solver.bdf1_step(dt, None, euler_forcing, problem)
    sol = solver.bdf2_step(final_time, None, euler_forcing, problem)

    u_exact = velocity_exact(final_time, xu)
    xp = dual.get_pressure_domain().get_int_bdry_nodes()
    p_exact = pressure_exact(xp)
    pressure = sol["pressure"] - np.mean(sol["pressure"] - p_exact)

    result = CaseResult(
        xi_u=xi_u,
        xi_p=xi_p,
        h=h,
        nu=xu.shape[0],
        np_=xp.shape[0],
        u_rel=relative_l2(sol["velocity"], u_exact),
        p_rel=relative_l2(pressure, p_exact),
        div_rms=float(sol["divergence_rms"]),
        div_max=float(sol["divergence_max"]),
        wall_rms=float(sol["wall_normal_rms"]),
        wall_max=float(sol["wall_normal_max"]),
    )
    fields = {
        "xu": xu,
        "u_exact": u_exact,
        "u_num": np.asarray(sol["velocity"], dtype=float),
    }
    return result, fields


def run_study(xi_u: int = 4, xi_p: int | None = None, hvals: list[float] | None = None, dt: float = 0.02, final_time: float = 0.04):
    if xi_p is None:
        xi_p = xi_u
    if hvals is None:
        hvals = [0.14, 0.10, 0.08]
    results: list[CaseResult] = []
    finest_fields: dict[str, np.ndarray] | None = None
    for h in hvals:
        case_result, fields = run_case(xi_u, xi_p, h, dt, final_time)
        results.append(case_result)
        finest_fields = fields
        print(
            f"h={case_result.h:.3f}  Nu={case_result.nu}  Np={case_result.np_}  "
            f"u_rel={case_result.u_rel:.6e}  p_rel={case_result.p_rel:.6e}  "
            f"div_rms={case_result.div_rms:.6e}  wall_rms={case_result.wall_rms:.6e}"
        )

    rates = [float("nan")]
    for k in range(1, len(results)):
        prev = results[k - 1]
        curr = results[k]
        rates.append(float(np.log(prev.u_rel / curr.u_rel) / np.log(prev.h / curr.h)))
    return results, rates, finest_fields


def save_outputs(results: list[CaseResult], rates: list[float], finest_fields: dict[str, np.ndarray] | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [asdict(r) for r in results],
        "rates": rates,
    }
    (out_dir / "pusl_incompressible_euler_convergence_2d.json").write_text(json.dumps(payload, indent=2))

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    hvals = [r.h for r in results]
    errs = [r.u_rel for r in results]
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.loglog(hvals, errs, "-o", lw=1.5, ms=7)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.set_xlabel("h")
    ax1.set_ylabel("relative velocity L2 error")
    ax1.set_title(f"PUSL Incompressible Euler on disk (xi_u = {results[0].xi_u}, xi_p = {results[0].xi_p})")
    fig1.tight_layout()
    fig1.savefig(out_dir / "pusl_incompressible_euler_convergence_2d.png", dpi=180)
    plt.close(fig1)

    if finest_fields is not None:
        fig2, axs = plt.subplots(1, 2, figsize=(10.5, 4.5))
        axs[0].quiver(finest_fields["xu"][:, 0], finest_fields["xu"][:, 1], finest_fields["u_exact"][:, 0], finest_fields["u_exact"][:, 1])
        axs[0].set_aspect("equal", adjustable="box")
        axs[0].grid(alpha=0.25)
        axs[0].set_title("Exact velocity")
        axs[1].quiver(finest_fields["xu"][:, 0], finest_fields["xu"][:, 1], finest_fields["u_num"][:, 0], finest_fields["u_num"][:, 1])
        axs[1].set_aspect("equal", adjustable="box")
        axs[1].grid(alpha=0.25)
        axs[1].set_title("Numerical velocity")
        fig2.tight_layout()
        fig2.savefig(out_dir / "pusl_incompressible_euler_velocity_fields_2d.png", dpi=180)
        plt.close(fig2)


def main() -> None:
    results, rates, finest_fields = run_study()
    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "convergence"
    save_outputs(results, rates, finest_fields, out_dir)
    print("PUSL incompressible Euler convergence study")
    for idx, result in enumerate(results):
        rate_text = "-" if idx == 0 or not np.isfinite(rates[idx]) else f"{rates[idx]:.4f}"
        print(
            f"  h={result.h:.3f}  Nu={result.nu}  Np={result.np_}  "
            f"u_rel={result.u_rel:.6e}  p_rel={result.p_rel:.6e}  rate={rate_text}"
        )


if __name__ == "__main__":
    main()
