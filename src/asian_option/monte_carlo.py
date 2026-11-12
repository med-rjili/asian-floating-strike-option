"""Monte Carlo pricing for the arithmetic floating-strike Asian put."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    n_paths: int
    n_steps: int


def price_monte_carlo(
    spot: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int = 200_000,
    n_steps: int = 500,
    seed: int | None = 42,
    confidence_z: float = 1.959963984540054,
    chunk_size: int = 20_000,
) -> MonteCarloResult:
    """Estimate exp(-rT) E[(A_T-S_T)+] under risk-neutral GBM.

    The time integral is approximated by the composite trapezoidal rule.
    Paths are simulated in chunks to bound memory consumption.
    """
    if spot <= 0 or maturity <= 0 or volatility <= 0:
        raise ValueError("spot, maturity and volatility must be positive")
    if n_paths < 2 or n_steps < 1 or chunk_size < 1:
        raise ValueError("n_paths >= 2, n_steps >= 1 and chunk_size >= 1 required")
    if confidence_z <= 0:
        raise ValueError("confidence_z must be positive")

    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    drift = (rate - 0.5 * volatility**2) * dt
    diffusion = volatility * np.sqrt(dt)
    discount = np.exp(-rate * maturity)

    payoff_sum = 0.0
    payoff_square_sum = 0.0
    generated = 0
    while generated < n_paths:
        count = min(chunk_size, n_paths - generated)
        current = np.full(count, spot, dtype=float)
        integral = np.zeros(count, dtype=float)
        for _ in range(n_steps):
            z = rng.standard_normal(count)
            nxt = current * np.exp(drift + diffusion * z)
            integral += 0.5 * (current + nxt) * dt
            current = nxt
        discounted_payoff = discount * np.maximum(integral / maturity - current, 0.0)
        payoff_sum += float(np.sum(discounted_payoff))
        payoff_square_sum += float(np.dot(discounted_payoff, discounted_payoff))
        generated += count

    mean = payoff_sum / n_paths
    centered_sum_squares = max(payoff_square_sum - n_paths * mean**2, 0.0)
    sample_variance = centered_sum_squares / (n_paths - 1)
    standard_error = float(np.sqrt(sample_variance / n_paths))
    half_width = confidence_z * standard_error
    return MonteCarloResult(
        price=float(mean),
        standard_error=standard_error,
        confidence_interval=(float(mean - half_width), float(mean + half_width)),
        n_paths=n_paths,
        n_steps=n_steps,
    )
