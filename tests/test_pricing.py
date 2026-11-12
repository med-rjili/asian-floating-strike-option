import numpy as np
import pytest

from asian_option.finite_difference import _left_boundary, solve_pde
from asian_option.monte_carlo import price_monte_carlo


def test_initial_condition_is_floating_put_payoff():
    result = solve_pde(
        maturity=1.0, rate=0.03, volatility=0.2,
        x_min=-6.0, x_max=3.0, n_space=101, n_time=10, theta=1.0
    )
    np.testing.assert_allclose(result.values[0], np.maximum(-1.0 - result.x, 0.0))


def test_affine_left_boundary_has_correct_zero_rate_limit():
    tau, x_min, maturity = 0.4, -10.0, 1.0
    got = _left_boundary(tau, x_min, maturity, 0.0)
    assert got == pytest.approx(-x_min + tau / maturity - 1.0)


def test_pde_solution_is_finite_and_price_nonnegative():
    result = solve_pde(
        maturity=1.0, rate=0.03, volatility=0.25,
        x_min=-8.0, x_max=4.0, n_space=301, n_time=400, theta=1.0
    )
    assert np.isfinite(result.price(100.0))
    assert result.price(100.0) >= -1e-8


def test_monte_carlo_reproducibility_and_confidence_interval():
    kwargs = dict(
        spot=100.0, maturity=1.0, rate=0.03, volatility=0.2,
        n_paths=4000, n_steps=50, seed=123, chunk_size=1000
    )
    first = price_monte_carlo(**kwargs)
    second = price_monte_carlo(**kwargs)
    assert first.price == pytest.approx(second.price)
    low, high = first.confidence_interval
    assert low <= first.price <= high
    assert first.standard_error >= 0.0


def test_invalid_spatial_domain_is_rejected():
    with pytest.raises(ValueError):
        solve_pde(
            maturity=1.0, rate=0.03, volatility=0.2,
            x_min=0.0, x_max=4.0, n_space=101, n_time=10
        )
