import numpy as np
import pandas as pd
import pytest

from uplift_causal.evaluate_uplift import (
    budget_table,
    diff_means,
    qini_ci,
    qini_coefficient,
    qini_curve,
    segments,
    uplift_band,
)
from uplift_causal.simulate_bias import simulate_bias


@pytest.fixture(scope="module")
def rct():
    rng = np.random.default_rng(0)
    n = 200_000
    x = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    tau = 0.05 + 0.05 * (x > 0)
    y = (rng.random(n) < 0.1 + 0.1 * (x > 0) + tau * t).astype(int)
    return pd.DataFrame({"x": x, "t": t, "y": y}), tau


def test_diff_means_covers_truth(rct):
    df, tau = rct
    _, lo, hi = diff_means(df.y, df.t)
    assert lo < tau.mean() < hi


def test_bias_preserves_covariates_and_biases_naive(rct):
    df, tau = rct
    obs = simulate_bias(df, {"x": 1.0}, strength=2.0)
    assert abs(obs.x.mean() - df.x.mean()) < 0.02
    assert obs.groupby("t").x.mean().diff().iloc[-1] > 0.5
    assert diff_means(obs.y, obs.t)[1] > tau.mean()


def test_qini_oracle_beats_random(rct):
    df, tau = rct
    x, q = qini_curve(df.y, df.t, tau)
    assert x[0] == 0 and q[0] == 0 and x[-1] == 1
    assert qini_coefficient(df.y, df.t, tau) > 0
    assert qini_coefficient(df.y, df.t, tau) > qini_coefficient(df.y, df.t, -tau)


def test_budget_table_endpoints(rct):
    df, tau = rct
    b = budget_table(df.y, df.t, tau, fractions=(0.5, 1.0), n_boot=20)
    assert b.effect_share.iloc[-1] == pytest.approx(1.0)
    assert b.effect_share.iloc[0] > 0.5
    assert (b.lo <= b.effect_share).all() and (b.effect_share <= b.hi).all()


def test_qini_normalized_bounds(rct):
    df, tau = rct
    q, lo, hi = qini_ci(df.y, df.t, tau, n_boot=20)
    assert 0 < lo <= q <= hi < 1


def test_segments_rule():
    seg = segments([0.1, 0.3, 0.1, 0.3, 0.3], [0.0, 0.0, 0.05, -0.05, 0.05], 0.2, 0.02)
    assert list(seg) == ["Lost Causes", "Sure Things", "Persuadables", "Sleeping Dogs", "Persuadables"]


def test_uplift_band_ends_at_ate(rct):
    df, tau = rct
    grid, est, lo, hi = uplift_band(df.y, df.t, tau, n_boot=20)
    assert est[0] == 0 and grid[-1] == 1
    assert est[-1] == pytest.approx(diff_means(df.y, df.t)[0])
    assert (lo <= est).all() and (est <= hi).all()
