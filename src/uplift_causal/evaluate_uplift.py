import numpy as np
import pandas as pd
from scipy import stats

SEGMENTS = ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]


def diff_means(y, t, alpha=0.05):
    y, t = np.asarray(y, dtype=float), np.asarray(t).astype(bool)
    y1, y0 = y[t], y[~t]
    est = y1.mean() - y0.mean()
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    se = np.sqrt(v1 + v0)
    dof = (v1 + v0) ** 2 / (v1**2 / (len(y1) - 1) + v0**2 / (len(y0) - 1)) if se > 0 else 1.0
    z = stats.t.ppf(1 - alpha / 2, dof)
    return est, est - z * se, est + z * se


def _sorted(y, t, score):
    o = np.argsort(-np.asarray(score), kind="stable")
    return np.asarray(y, dtype=float)[o], np.asarray(t, dtype=float)[o]


def _ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


def _cum(y, t, w):
    n = np.cumsum(w)
    nt = np.cumsum(w * t)
    return n, nt, n - nt, np.cumsum(w * y * t), np.cumsum(w * y * (1 - t))


def _qini(y, t, w):
    n, nt, nc, yt, yc = _cum(y, t, w)
    return n / n[-1], yt - yc * _ratio(nt, nc)


def _uplift(y, t, w):
    n, nt, nc, yt, yc = _cum(y, t, w)
    return n / n[-1], (_ratio(yt, nt) - _ratio(yc, nc)) * n


def _area(y, t, w):
    x, q = _qini(y, t, w)
    return np.sum(w * (q - x * q[-1])) / w.sum() ** 2


def _bootstrap(stat, n, n_boot, alpha, seed):
    rng = np.random.default_rng(seed)
    boot = np.array([stat(rng.poisson(1.0, n).astype(float)) for _ in range(n_boot)])
    return stat(np.ones(n)), *np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)


def _qini_stat(y, t, score, normalize):
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    scores = [score, (2 * t - 1) * y] if normalize else [score]
    parts = [(y[o], t[o], o) for o in (np.argsort(-np.asarray(s), kind="stable") for s in scores)]

    def stat(w):
        a = [_area(ys, ts, w[o]) for ys, ts, o in parts]
        return a[0] / a[1] if normalize else a[0]

    return stat


def qini_curve(y, t, score):
    x, q = _qini(*_sorted(y, t, score), np.ones(len(y)))
    return np.r_[0.0, x], np.r_[0.0, q]


def uplift_curve(y, t, score):
    x, u = _uplift(*_sorted(y, t, score), np.ones(len(y)))
    return np.r_[0.0, x], np.r_[0.0, u]


def qini_coefficient(y, t, score, normalize=True):
    return _qini_stat(y, t, score, normalize)(np.ones(len(y)))


def qini_ci(y, t, score, normalize=True, n_boot=200, alpha=0.05, seed=0):
    return _bootstrap(_qini_stat(y, t, score, normalize), len(y), n_boot, alpha, seed)


def qini_diff_ci(y, t, score_a, score_b, normalize=True, n_boot=200, alpha=0.05, seed=0):
    a, b = _qini_stat(y, t, score_a, normalize), _qini_stat(y, t, score_b, normalize)
    return _bootstrap(lambda w: a(w) - b(w), len(y), n_boot, alpha, seed)


def _gamma(y, t, p=None, m=None):
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    p = t.mean() if p is None else p
    return (y - (0.0 if m is None else np.asarray(m))) * (t - p) / (p * (1 - p))


def rate(y, t, score, p=None, m=None, target="autoc", n_boot=200, seed=0):
    g = _gamma(y, t, p, m)[np.argsort(-np.asarray(score), kind="stable")]

    def stat(w):
        n = np.cumsum(w)
        toc = _ratio(np.cumsum(w * g), n) - np.sum(w * g) / n[-1]
        return np.sum(w * toc * (n / n[-1] if target == "qini" else 1.0)) / n[-1]

    rng = np.random.default_rng(seed)
    est = stat(np.ones(len(g)))
    se = np.std([stat(rng.poisson(1.0, len(g)).astype(float)) for _ in range(n_boot)], ddof=1)
    return est, se, 2 * stats.norm.sf(abs(est / se))


def blp(y, t, tau, mu0=None, p=None):
    y, t, tau = np.asarray(y, dtype=float), np.asarray(t, dtype=float), np.asarray(tau)
    p = t.mean() if p is None else p
    base = [np.ones(len(y))] if mu0 is None else [np.ones(len(y)), mu0]
    X = np.column_stack([*base, t - p, (t - p) * (tau - tau.mean())])
    xtx = np.linalg.pinv(X.T @ X)
    beta = xtx @ X.T @ y
    r = y - X @ beta
    cov = xtx @ (X.T * r**2) @ X @ xtx * len(y) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(cov))
    out = pd.DataFrame({"coef": beta, "se": se, "p_value": 2 * stats.norm.sf(np.abs(beta / se))})
    out = out.iloc[-2:]
    out.index = ["ate", "heterogeneity"]
    return out


def gates(y, t, tau, groups=5):
    rank = stats.rankdata(tau, method="ordinal")
    q = np.minimum((rank - 1) * groups // len(rank), groups - 1) + 1
    df = pd.DataFrame({"y": np.asarray(y), "t": np.asarray(t), "tau": np.asarray(tau), "group": q})
    rows = {
        k: (g.tau.mean(), *diff_means(g.y, g.t)) for k, g in df.groupby("group") if g.t.nunique() == 2
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["predicted", "uplift", "lo", "hi"])


def benjamini_hochberg(p_values, q=0.1):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, len(p) + 1) / len(p)
    reject = np.zeros(len(p), dtype=bool)
    if passed.any():
        reject[order[: np.flatnonzero(passed).max() + 1]] = True
    return reject


def segments_fdr(mu0, tau, p_values, threshold, q=0.1):
    mu0, tau = np.asarray(mu0), np.asarray(tau)
    found = benjamini_hochberg(p_values, q)
    return np.select(
        [found & (tau > 0), found & (tau < 0), mu0 >= threshold],
        ["Persuadables", "Sleeping Dogs", "Sure Things"],
        "Lost Causes",
    )


def effective_sample_size(w):
    w = np.asarray(w, dtype=float)
    return w.sum() ** 2 / (w**2).sum()


def segments(mu0, tau, threshold, delta):
    mu0, tau = np.asarray(mu0), np.asarray(tau)
    return np.select(
        [tau > delta, tau < -delta, mu0 >= threshold],
        ["Persuadables", "Sleeping Dogs", "Sure Things"],
        "Lost Causes",
    )


def segment_table(y, t, seg):
    df = pd.DataFrame({"y": y, "t": t, "seg": seg})
    rows = {
        s: (len(g) / len(df), *diff_means(g.y, g.t))
        for s, g in df.groupby("seg")
        if g.t.nunique() == 2
    }
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["share", "uplift", "lo", "hi"])
    return out.reindex([s for s in SEGMENTS if s in out.index])


def budget_table(
    y, t, score, fractions=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0), n_boot=200, alpha=0.05, seed=0
):
    ys, ts = _sorted(y, t, score)
    f = np.asarray(fractions)

    def stat(w):
        x, u = _uplift(ys, ts, w)
        return u[np.minimum(np.searchsorted(x, f), len(x) - 1)] / u[-1]

    est, lo, hi = _bootstrap(stat, len(ys), n_boot, alpha, seed)
    return pd.DataFrame({"cost_share": f, "effect_share": est, "lo": lo, "hi": hi})


def uplift_band(y, t, score, points=101, n_boot=200, alpha=0.05, seed=0):
    ys, ts = _sorted(y, t, score)
    grid = np.linspace(0, 1, points)

    def stat(w):
        x, u = _uplift(ys, ts, w)
        return np.r_[0.0, u[np.minimum(np.searchsorted(x, grid[1:]), len(x) - 1)] / w.sum()]

    return grid, *_bootstrap(stat, len(ys), n_boot, alpha, seed)


def uplift_oob(y, t, score, points=101, n_boot=200, seed=0):
    ys, ts = _sorted(y, t, score)
    grid = np.linspace(0, 1, points)
    rng = np.random.default_rng(seed)

    def curve(w):
        x, u = _uplift(ys, ts, w)
        return np.r_[0.0, u[np.minimum(np.searchsorted(x, grid[1:]), len(x) - 1)] / w.sum()]

    bags = [rng.poisson(1.0, len(ys)).astype(float) for _ in range(n_boot)]
    return grid, np.array([curve(w) for w in bags]), np.array([curve((w == 0) * 1.0) for w in bags])


def honest_optimum(grid, inbag, oob, value, cost, max_share=1.0, alpha=0.05):
    allowed = grid <= max_share + 1e-12
    pick = np.argmax(np.where(allowed, value * inbag - cost * grid, -np.inf), axis=1)
    rows = np.arange(len(pick))
    honest = value * oob[rows, pick] - cost * grid[pick]
    return {
        "naive": float((value * inbag[rows, pick] - cost * grid[pick]).mean()),
        "honest": float(honest.mean()),
        "share": tuple(np.quantile(grid[pick], [alpha / 2, 0.5, 1 - alpha / 2])),
    }
