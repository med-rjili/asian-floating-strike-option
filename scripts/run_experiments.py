"""Run PDE/Monte Carlo comparisons and save diagnostic figures in results/."""

from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asian_option.finite_difference import solve_pde  # noqa: E402
from asian_option.monte_carlo import price_monte_carlo  # noqa: E402


def main() -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    spot, rate, volatility, maturity = 100.0, 0.03, 0.25, 1.0

    mc = price_monte_carlo(
        spot, maturity, rate, volatility,
        n_paths=250_000, n_steps=800, seed=2026
    )
    cn = solve_pde(
        maturity, rate, volatility, x_min=-10.0, x_max=4.0,
        n_space=1001, n_time=1600, theta=0.5
    )
    implicit = solve_pde(
        maturity, rate, volatility, x_min=-10.0, x_max=4.0,
        n_space=1001, n_time=1600, theta=1.0
    )
    cn_price, implicit_price = cn.price(spot), implicit.price(spot)

    labels = ["Crank-Nicolson", "Implicit Euler", "Monte Carlo"]
    estimates = [cn_price, implicit_price, mc.price]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, estimates)
    ax.errorbar(
        [2], [mc.price],
        yerr=[[mc.price - mc.confidence_interval[0]],
              [mc.confidence_interval[1] - mc.price]],
        fmt="none", capsize=5, color="black"
    )
    ax.set_ylabel("Option price")
    ax.set_title("Floating-strike Asian put: PDE vs Monte Carlo")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "pde_mc_comparison.png", dpi=160)
    plt.close(fig)

    space_sizes = [201, 301, 501, 701, 1001]
    space_prices = []
    for n_space in space_sizes:
        sol = solve_pde(
            maturity, rate, volatility, x_min=-10.0, x_max=4.0,
            n_space=n_space, n_time=1800, theta=0.5
        )
        space_prices.append(sol.price(spot))
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(space_sizes, space_prices, marker="o")
    ax.set_xlabel("Number of spatial grid points")
    ax.set_ylabel("PDE price")
    ax.set_title("Spatial-grid refinement (Crank-Nicolson)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "spatial_convergence.png", dpi=160)
    plt.close(fig)

    time_sizes = [200, 400, 800, 1600, 3200]
    time_prices = []
    for n_time in time_sizes:
        sol = solve_pde(
            maturity, rate, volatility, x_min=-10.0, x_max=4.0,
            n_space=801, n_time=n_time, theta=0.5
        )
        time_prices.append(sol.price(spot))
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(time_sizes, time_prices, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of time steps")
    ax.set_ylabel("PDE price")
    ax.set_title("Time-step refinement (Crank-Nicolson)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "temporal_convergence.png", dpi=160)
    plt.close(fig)

    summary = {
        "parameters": {"spot": spot, "rate": rate, "volatility": volatility, "maturity": maturity},
        "crank_nicolson_price": cn_price,
        "implicit_euler_price": implicit_price,
        "monte_carlo_price": mc.price,
        "monte_carlo_standard_error": mc.standard_error,
        "monte_carlo_95_percent_confidence_interval": list(mc.confidence_interval),
        "monte_carlo_paths": mc.n_paths,
        "monte_carlo_time_steps": mc.n_steps,
        "spatial_refinement": {"n_space": space_sizes, "prices": space_prices},
        "temporal_refinement": {"n_time": time_sizes, "prices": time_prices},
        "note": "Check PDE domain truncation and Monte Carlo time-integration bias separately."
    }
    (results_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
