import numpy as np
import pandas as pd
from dowhy import CausalModel
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from .evaluate_uplift import diff_means
from .learners import lgbm

REFUTERS = ("random_common_cause", "placebo_treatment_refuter", "data_subset_refuter")


def causal_model(df):
    common = [c for c in df if c not in ("t", "y")]
    model = CausalModel(data=df.copy(), treatment="t", outcome="y", common_causes=common)
    return model, model.identify_effect(proceed_when_unidentifiable=True)


def estimate(df, method, confidence_intervals=False, n_boot=100, seed=0):
    m, p = methods(n_boot, seed)[method]
    model, estimand = causal_model(df)
    return estimand, model.estimate_effect(
        estimand, method_name=m, method_params=p, confidence_intervals=confidence_intervals
    )


def methods(n_boot=100, seed=0):
    boot = {"num_simulations": n_boot, "random_state": seed}
    return {
        "PS Matching": ("backdoor.propensity_score_matching", {"init_params": boot}),
        "IPW": (
            "backdoor.propensity_score_weighting",
            {"init_params": boot | {"weighting_scheme": "ips_normalized_weight"}},
        ),
        "DML": (
            "backdoor.econml.dml.LinearDML",
            {
                "init_params": {
                    "model_y": lgbm(seed=seed),
                    "model_t": lgbm(LGBMClassifier, seed=seed),
                    "discrete_treatment": True,
                    "cv": 3,
                    "random_state": seed,
                },
                "fit_params": {},
            },
        ),
    }


def fit_estimators(df, n_boot=100, seed=0):
    return {name: estimate(df, name, True, n_boot, seed)[1] for name in methods()}


def summarize(estimates):
    rows = {}
    for name, e in estimates.items():
        ci = np.ravel(e.get_confidence_intervals())
        rows[name] = (float(np.ravel(e.value)[0]), ci[0], ci[-1])
    return pd.DataFrame.from_dict(rows, orient="index", columns=["est", "lo", "hi"])


def aipw(df, cv=3, seed=0, clip=0.01, alpha=0.05, n_estimators=100):
    X, t, y = df.drop(columns=["t", "y"]), df.t.to_numpy(), df.y.to_numpy(dtype=float)
    e, mu = np.empty(len(df)), np.empty((2, len(df)))
    for train, test in StratifiedKFold(cv, shuffle=True, random_state=seed).split(X, t):
        model = lgbm(LGBMClassifier, seed=seed, n_estimators=n_estimators).fit(X.iloc[train], t[train])
        e[test] = model.predict_proba(X.iloc[test])[:, 1]
        for arm in (0, 1):
            idx = train[t[train] == arm]
            model = lgbm(seed=seed, n_estimators=n_estimators).fit(X.iloc[idx], y[idx])
            mu[arm, test] = model.predict(X.iloc[test])
    e = np.clip(e, clip, 1 - clip)
    psi = mu[1] - mu[0] + t * (y - mu[1]) / e - (1 - t) * (y - mu[0]) / (1 - e)
    est, se = psi.mean(), psi.std(ddof=1) / np.sqrt(len(psi))
    z = stats.norm.ppf(1 - alpha / 2)
    return est, est - z * se, est + z * se


def compare(df, n_boot=100, seed=0):
    rows = {"Naive": diff_means(df.y, df.t)}
    rows |= {k: tuple(v) for k, v in summarize(fit_estimators(df, n_boot, seed)).iterrows()}
    rows["AIPW"] = aipw(df, seed=seed)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["est", "lo", "hi"])


def refute(df, method="IPW", n_sim=50, seed=0):
    estimand, est = estimate(df, method, seed=seed)
    model, _ = causal_model(df)
    rows = {}
    for r in REFUTERS:
        res = model.refute_estimate(
            estimand, est, method_name=r, num_simulations=n_sim, random_seed=seed
        )
        rows[r] = (res.estimated_effect, res.new_effect, (res.refutation_result or {}).get("p_value"))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["estimate", "refuted", "p_value"])
