"""Finite-difference solver for a floating-strike arithmetic Asian put."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class FDSolution:
    """PDE grid and solution, indexed as values[time_index, space_index]."""
    x: np.ndarray
    tau: np.ndarray
    values: np.ndarray
    theta: float

    def reduced_value_at_zero(self) -> float:
        """Return f(T, 0), the dimensionless time-zero price."""
        return float(np.interp(0.0, self.x, self.values[-1]))

    def price(self, spot: float) -> float:
        """Return the time-zero option value for a positive spot."""
        if spot <= 0:
            raise ValueError("spot must be strictly positive")
        return spot * self.reduced_value_at_zero()


def _left_boundary(tau: float, x_min: float, maturity: float, rate: float) -> float:
    """Affine asymptotic boundary, exact on the linear payoff branch."""
    if abs(rate) < 1e-12:
        integral_term = tau / maturity
    else:
        integral_term = -np.expm1(-rate * tau) / (rate * maturity)
    return -np.exp(-rate * tau) * x_min + integral_term - 1.0


def solve_pde(
    maturity: float,
    rate: float,
    volatility: float,
    x_min: float = -8.0,
    x_max: float = 4.0,
    n_space: int = 801,
    n_time: int = 1200,
    theta: float = 0.5,
) -> FDSolution:
    """Solve the reduced PDE with centered space differences and a theta scheme.

    theta=0: explicit Euler; theta=0.5: Crank-Nicolson;
    theta=1: implicit Euler. Explicit Euler is conditionally stable.
    """
    if maturity <= 0 or volatility <= 0:
        raise ValueError("maturity and volatility must be positive")
    if n_space < 5 or n_time < 1:
        raise ValueError("n_space must be >= 5 and n_time must be >= 1")
    if not (x_min < 0.0 < x_max):
        raise ValueError("the spatial domain must satisfy x_min < 0 < x_max")
    if not (0.0 <= theta <= 1.0):
        raise ValueError("theta must lie in [0, 1]")

    x = np.linspace(x_min, x_max, n_space, dtype=float)
    dx = x[1] - x[0]
    dt = maturity / n_time
    tau = np.linspace(0.0, maturity, n_time + 1)

    xi = x[1:-1]
    diffusion = 0.5 * volatility**2 * xi**2
    drift = -(1.0 / maturity + rate * xi)
    lower = diffusion / dx**2 - drift / (2.0 * dx)
    diagonal = -2.0 * diffusion / dx**2
    upper = diffusion / dx**2 + drift / (2.0 * dx)

    operator = diags(
        [lower[1:], diagonal, upper[:-1]],
        offsets=[-1, 0, 1],
        shape=(n_space - 2, n_space - 2),
        format="csc",
    )
    identity = eye(n_space - 2, format="csc")
    lhs = (identity - theta * dt * operator).tocsc()
    rhs_operator = (identity + (1.0 - theta) * dt * operator).tocsc()
    factor = splu(lhs) if theta > 0.0 else None

    values = np.empty((n_time + 1, n_space), dtype=float)
    values[0] = np.maximum(-1.0 - x, 0.0)

    def boundary_vector(time_to_maturity: float) -> np.ndarray:
        result = np.zeros(n_space - 2, dtype=float)
        result[0] = lower[0] * _left_boundary(
            time_to_maturity, x_min, maturity, rate
        )
        # Right boundary is the zero asymptote at sufficiently large x.
        result[-1] = 0.0
        return result

    for n in range(n_time):
        old_boundary = boundary_vector(tau[n])
        new_boundary = boundary_vector(tau[n + 1])
        rhs = rhs_operator @ values[n, 1:-1]
        rhs += dt * ((1.0 - theta) * old_boundary + theta * new_boundary)
        next_interior = rhs if factor is None else factor.solve(rhs)

        values[n + 1, 0] = _left_boundary(
            tau[n + 1], x_min, maturity, rate
        )
        values[n + 1, -1] = 0.0
        values[n + 1, 1:-1] = next_interior

    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite PDE values; refine the grid/time step")
    return FDSolution(x=x, tau=tau, values=values, theta=float(theta))
