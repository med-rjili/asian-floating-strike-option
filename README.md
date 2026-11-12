# Asian Floating-Strike Put Option

A reproducible Python project for pricing an arithmetic-average floating-strike Asian put under Black–Scholes. It includes a reduced one-dimensional PDE solver, an independent Monte Carlo benchmark, tests, and scripts for diagnostic plots.

**Validation status:** the source files are being committed, but numerical experiments have not been executed in this session. No numerical output should be treated as validated until the tests and experiment script have been run.

## Contract

Under the risk-neutral measure:
$$

$$
dS_t=rS_t\,dt+\sigma S_t\,dW_t,\qquad S_0>0.
$$

$$
The arithmetic average and floating-strike put payoff are
$$

$$
A_T=\frac1T\int_0^T S_u\,du,\qquad (A_T-S_T)^+.
$$

$$
The time-zero value is
$$

$$
V_0=e^{-rT}\mathbb E^\mathbb Q[(A_T-S_T)^+].
$$

$$

## PDE reduction

Use homogeneity and the transformed state
$$

$$
x=-\frac{t}{T}\frac{A_t}{S_t},\qquad V(t,S,A)=S f(T-t,x).
$$

$$
With time-to-maturity $\tau=T-t$, the reduced PDE is
$$

$$
\partial_\tau f
=\frac12\sigma^2x^2\partial_{xx}f
-\left(\frac1T+rx\right)\partial_xf,
\qquad f(0,x)=(-1-x)^+.
$$

$$
Recover the time-zero price from
$$

$$
V_0=S_0 f(T,0).
$$

$$

The solver uses a truncated interval [x_min, x_max], with x_min strictly negative. The left boundary is the affine asymptotic solution
$$

$$
f(\tau,x_{\min})=-e^{-r\tau}x_{\min}
+\frac{1-e^{-r\tau}}{rT}-1,
$$

$$
and the right boundary is set to zero. For r=0, the fraction has limit tau/T.

**Domain warning:** the transformed state is non-positive in the original model. Setting x_min=0 removes the negative states and is not a suitable production pricing domain. Check sensitivity by moving x_min further left and increasing x_max.

## Numerical methods

Centered finite differences are used in space with a theta scheme in time:
- Explicit Euler: theta=0; conditionally stable and requires a time-step stability study.
- Crank–Nicolson: theta=1/2; formally second-order in time for sufficiently smooth solutions. The payoff kink may reduce observed order on coarse grids.
- Implicit Euler: theta=1; first-order in time and more damping.

The spatial operator is sparse and tridiagonal. The implicit matrix is factorized once and reused.

The Monte Carlo benchmark simulates risk-neutral geometric Brownian paths and approximates the continuous time integral with the composite trapezoidal rule. It reports a sampling standard error and normal-approximation 95% confidence interval. This interval does not account for time-discretization bias in the path integral.

## Installation

Python 3.10+ is recommended.

~~~bash
git clone https://github.com/med-rjili/asian-floating-strike-option.git
cd asian-floating-strike-option
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
~~~

## Quick start

~~~python
from asian_option.finite_difference import solve_pde
from asian_option.monte_carlo import price_monte_carlo

S0, r, sigma, T = 100.0, 0.03, 0.25, 1.0
fd = solve_pde(T, r, sigma, x_min=-8, x_max=4,
               n_space=801, n_time=1200, theta=0.5)
print("PDE price:", fd.price(S0))

mc = price_monte_carlo(S0, T, r, sigma, n_paths=200_000,
                       n_steps=500, seed=42)
print("MC estimate:", mc.price)
print("95% CI:", mc.confidence_interval)
~~~

## Run tests and experiments

~~~bash
pytest -q
python scripts/run_experiments.py
~~~

The editable install above makes the src-layout package importable from the examples and tests.

The experiment script writes the following to results/:
- PDE versus Monte Carlo comparison with Monte Carlo confidence interval;
- spatial-grid refinement plot;
- time-step refinement plot;
- JSON summary of parameters and estimates.

The script is provided to generate the figures; they are not committed as precomputed results because they have not yet been executed in this environment.

## Validation checklist

1. Run the unit tests.
2. Compare implicit Euler and Crank–Nicolson on successively refined grids.
3. Increase the magnitude of x_min and x_max to check boundary truncation sensitivity.
4. Increase Monte Carlo path count and time steps separately.
5. Compare the PDE estimate with the Monte Carlo confidence interval, allowing for PDE discretization and Monte Carlo integration bias.
6. Repeat with several random seeds. One confidence interval is not proof of correctness.

## Assumptions and scope

- European-style payoff evaluated at maturity.
- Constant risk-free rate and volatility.
- No dividends, transaction costs, or market frictions.
- Continuous arithmetic averaging in the contract; Monte Carlo approximates the integral numerically.
- PDE boundaries are truncations and must be sensitivity-tested.

## References

- Kemna, A. G. Z. and Vorst, A. C. F. (1990), “A pricing method for options based on average asset values”, Journal of Banking & Finance.
- Hull, J. C., Options, Futures, and Other Derivatives, sections on Asian options and risk-neutral valuation.
