import numpy as np
import pandas as pd
from scipy import stats

SEGMENTS = ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]


def diff_means(y, t, alpha=0.05):
    y, t = np.asarray(y, dtype=float), np.asarray(t).astype(bool)
    y1, y0 = y[t], y[~t]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = stats.norm.ppf(1 - alpha / 2)
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
