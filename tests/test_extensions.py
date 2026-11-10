import numpy as np
import pandas as pd
import pytest
from scipy import stats

from uplift_causal.evaluate_uplift import (
    benjamini_hochberg,
    blp,
    diff_means,
    effective_sample_size,
    gates,
    honest_optimum,
    qini_coefficient,
    qini_diff_ci,
    rate,
    segments_fdr,
    uplift_band,
    uplift_oob,
)
from uplift_causal.montecarlo import summarize
from uplift_causal.sensitivity import e_value, risk_ratio, robustness_value
from uplift_causal.simulate_bias import nonlinear_propensity, simulate_bias


@pytest.fixture(scope="module")
def rct():
    rng = np.random.default_rng(0)
    n = 200_000
    x = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    tau = 0.05 + 0.05 * (x > 0)
    mu0 = 0.1 + 0.1 * (x > 0)
    y = (rng.random(n) < mu0 + tau * t).astype(int)
    return pd.DataFrame({"x": x, "t": t, "y": y}), tau, mu0


def flat(n, seed):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.1 + 0.05 * t).astype(int)
    return y, t, rng.normal(size=n)


def test_diff_means_matches_welch(rct):
    df, _, _ = rct
    d = df.head(60)
    ci = stats.ttest_ind(d.y[d.t == 1], d.y[d.t == 0], equal_var=False).confidence_interval()
    est, lo, hi = diff_means(d.y, d.t)
    assert (lo, hi) == pytest.approx((ci.low, ci.high))
    assert est == pytest.approx(d.y[d.t == 1].mean() - d.y[d.t == 0].mean())


def reference_qini(y, t, score):
    def area(s):
        o = np.argsort(-s, kind="stable")
        nt = nc = yt = yc = total = 0.0
        q = []
        for i in o:
            nt, nc = nt + t[i], nc + 1 - t[i]
            yt, yc = yt + y[i] * t[i], yc + y[i] * (1 - t[i])
            q.append(yt - (yc * nt / nc if nc else 0.0))
        n = len(y)
        for k, v in enumerate(q, 1):
            total += v - k / n * q[-1]
        return total / n**2

    return area(score) / area((2 * t - 1) * y)


def test_qini_matches_reference():
    rng = np.random.default_rng(1)
    n = 500
    t = rng.integers(0, 2, n).astype(float)
    score = rng.normal(size=n)
    y = (rng.random(n) < 0.2 + 0.2 * t * (score > 0)).astype(float)
    assert qini_coefficient(y, t, score) == pytest.approx(reference_qini(y, t, score))


def test_qini_diff_ci(rct):
    df, tau, _ = rct
    d, s = df.head(20_000), tau[:20_000]
    noise = np.random.default_rng(2).normal(size=len(d))
    est, lo, hi = qini_diff_ci(d.y, d.t, s, noise, n_boot=50)
    assert 0 < lo <= est <= hi
    assert qini_diff_ci(d.y, d.t, s, s, n_boot=10) == pytest.approx((0, 0, 0))


def test_rate_recovers_autoc(rct):
    df, tau, _ = rct
    est, se, p = rate(df.y, df.t, tau, p=0.5, n_boot=50)
    assert abs(est - 0.025 * np.log(2)) < 3 * se
    assert p < 1e-3
    qini = rate(df.y, df.t, tau, p=0.5, target="qini", n_boot=50)
    assert abs(qini[0] - 0.00625) < 3 * qini[1]


def test_rate_baseline_reduces_variance(rct):
    df, tau, mu0 = rct
    d, s, m = df.head(40_000), tau[:40_000], (mu0 + 0.5 * tau)[:40_000]
    assert rate(d.y, d.t, s, p=0.5, m=m, n_boot=50)[1] < rate(d.y, d.t, s, p=0.5, n_boot=50)[1]


def test_rate_is_calibrated_without_heterogeneity():
    p = np.array([rate(*flat(2000, seed), p=0.5, n_boot=100, seed=seed)[2] for seed in range(200)])
    assert 0.01 <= (p < 0.05).mean() <= 0.11


def test_blp_recovers_calibration(rct):
    df, tau, mu0 = rct
    out = blp(df.y, df.t, tau, mu0, p=0.5)
    assert abs(out.coef["ate"] - tau.mean()) < 3 * out.se["ate"]
    assert abs(out.coef["heterogeneity"] - 1) < 3 * out.se["heterogeneity"]
    assert out.p_value["heterogeneity"] < 1e-3
    half = blp(df.y, df.t, 2 * tau, mu0, p=0.5)
    assert half.coef["heterogeneity"] == pytest.approx(out.coef["heterogeneity"] / 2)
    plain = blp(df.y, df.t, tau, p=0.5)
    assert abs(plain.coef["heterogeneity"] - 1) < 3 * plain.se["heterogeneity"]
    assert plain.se["ate"] >= out.se["ate"]


def test_blp_finds_nothing_in_noise():
    p = []
    for seed in range(200):
        y, t, score = flat(2000, seed)
        p.append(blp(y, t, score, p=0.5).p_value["heterogeneity"])
    assert 0.01 <= (np.array(p) < 0.05).mean() <= 0.11


def test_gates_orders_groups(rct):
    df, tau, _ = rct
    g = gates(df.y, df.t, tau, groups=2)
    assert list(g.index) == [1, 2]
    assert g.predicted.tolist() == pytest.approx([0.05, 0.10], abs=1e-3)
    assert (g.lo < [0.05, 0.10]).all() and (g.hi > [0.05, 0.10]).all()
    assert len(gates(df.y, df.t, df.x, groups=5)) == 5


def test_benjamini_hochberg():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 8
    assert not benjamini_hochberg([0.5, 0.9], 0.1).any()
    assert benjamini_hochberg([0.9, 0.001], 0.1).tolist() == [False, True]


def test_segments_fdr():
    seg = segments_fdr([0.1, 0.3, 0.1, 0.3], [0.05, -0.05, 0.05, 0.0], [1e-4, 1e-4, 0.9, 0.9], 0.2)
    assert list(seg) == ["Persuadables", "Sleeping Dogs", "Lost Causes", "Sure Things"]


def test_effective_sample_size():
    assert effective_sample_size(np.ones(50)) == pytest.approx(50)
    assert effective_sample_size([1, 0, 0, 0]) == pytest.approx(1)


def test_oob_curves_match_band(rct):
    df, tau, _ = rct
    d, s = df.head(5000), tau[:5000]
    grid, inbag, oob = uplift_oob(d.y, d.t, s, points=11, n_boot=20)
    _, _, lo, hi = uplift_band(d.y, d.t, s, points=11, n_boot=20)
    assert inbag.shape == oob.shape == (20, 11)
    assert np.quantile(inbag, [0.025, 0.975], axis=0) == pytest.approx(np.array([lo, hi]))
    assert (inbag[:, 0] == 0).all() and (oob[:, 0] == 0).all()


def test_honest_optimum_removes_selection_bias():
    runs = []
    for seed in range(40):
        y, t, score = flat(5000, seed)
        runs.append(
            honest_optimum(*uplift_oob(y, t, score, points=21, n_boot=50), value=1.0, cost=0.05)
        )
    naive, honest = (np.array([r[k] for r in runs]) for k in ("naive", "honest"))
    se = honest.std(ddof=1) / np.sqrt(len(runs))
    assert naive.mean() > 3 * se
    assert abs(honest.mean()) < 3 * se
    assert (naive > honest).mean() > 0.9
    assert all(0 <= r["share"][0] <= r["share"][1] <= r["share"][2] <= 1 for r in runs)


def test_honest_optimum_respects_budget():
    y, t, score = flat(5000, 1)
    bags = uplift_oob(y, t, score, n_boot=20)
    assert honest_optimum(*bags, value=10.0, cost=0.0, max_share=0.3)["share"][2] <= 0.3
    assert honest_optimum(*bags, value=0.0, cost=1.0)["share"] == (0, 0, 0)


def confounded(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.1 + 0.1 * (a**2 > 1) + 0.05 * t).astype(int)
    return pd.DataFrame({"a": a, "b": b, "t": t, "y": y})


def test_nonlinear_bias_keeps_covariates_and_hides_from_linear_terms():
    df = confounded()
    weights = {"a": 1.0, "b": 1.0}
    e = nonlinear_propensity(df, weights, strength=1.5)
    assert e.min() >= 0.05 and e.max() <= 0.95
    obs = simulate_bias(df, weights, strength=1.5, propensity=nonlinear_propensity)
    assert abs(obs.a.mean() - df.a.mean()) < 0.03
    assert abs(obs.groupby("t").a.mean().diff().iloc[-1]) < 0.05
    assert (obs.a**2).groupby(obs.t).mean().diff().iloc[-1] > 0.5
    assert diff_means(obs.y, obs.t)[1] > 0.05


def test_aipw_covers_truth_under_nonlinear_confounding():
    from uplift_causal.estimate import aipw

    df = confounded()
    obs = simulate_bias(df, {"a": 1.0, "b": 1.0}, strength=1.5, propensity=nonlinear_propensity)
    est, lo, hi = aipw(obs)
    assert lo < 0.05 < hi
    assert hi - lo < 0.04


def test_sensitivity_values():
    assert e_value(1.0) == pytest.approx(1.0)
    assert e_value(2.0) == pytest.approx(2 + np.sqrt(2))
    assert e_value(0.5) == pytest.approx(e_value(2.0))
    assert risk_ratio([1, 1, 0, 1, 0, 0], [1, 1, 1, 0, 0, 0]) == pytest.approx(2.0)
    assert robustness_value(1.0, 0.1, 100) == pytest.approx((np.sqrt(5) - 1) / 2)
    assert robustness_value(1.0, 0.1, 100, q=0.5) < robustness_value(1.0, 0.1, 100)
    assert robustness_value(0.0, 0.1, 100) == 0


def test_montecarlo_summary():
    runs = pd.DataFrame(
        {
            "scenario": "s",
            "estimator": ["A"] * 4 + ["B"] * 4,
            "seed": [0, 1, 2, 3] * 2,
            "est": [0.9, 1.1, 0.9, 1.1, 2.0, 2.0, 2.0, 2.0],
            "lo": [0.5, 0.5, 0.5, 1.05, 1.5, 1.5, 1.5, 1.5],
            "hi": [1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5],
        }
    )
    out = summarize(runs, truth=1.0).loc["s"]
    assert out.bias.tolist() == pytest.approx([0.0, 1.0])
    assert out.rmse.tolist() == pytest.approx([0.1, 1.0])
    assert out.coverage.tolist() == pytest.approx([0.75, 0.0])
    assert out.reps.tolist() == [4, 4]


def test_cross_fit_gives_out_of_fold_scores_for_everyone():
    from uplift_causal.learners import cross_fit

    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    t = pd.Series(rng.integers(0, 2, n))
    y = pd.Series((rng.random(n) < 0.2 + 0.2 * t * (X.a > 0)).astype(int))
    names = ["S-Learner", "T-Learner", "DR-Learner"]
    out = cross_fit(X, t, y, names, n_splits=2, repeats=2, n_jobs=1)
    assert set(out) == set(names)
    for tau in out.values():
        assert tau.shape == (2, n) and np.isfinite(tau).all()
    assert not np.allclose(out["T-Learner"][0], out["T-Learner"][1])
    assert qini_coefficient(y, t, out["T-Learner"][0]) > 0


def test_compare_lists_all_estimators():
    from uplift_causal.estimate import compare

    df = confounded(8000)
    obs = simulate_bias(df, {"a": 1.0, "b": 1.0}, strength=1.5)
    out = compare(obs, n_boot=10)
    assert list(out.index) == ["Naive", "PS Matching", "IPW", "DML", "AIPW"]
    assert (out.lo <= out.est).all() and (out.est <= out.hi).all()
    assert out.est.between(-0.05, 0.15).all()


def test_app_runs():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=300).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Profit after selection correction" in labels and "Selection optimism" in labels
    at.number_input[1].set_value(1.0).run()
    assert not at.exception
