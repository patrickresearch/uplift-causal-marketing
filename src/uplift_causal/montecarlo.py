import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import learners
from .data import load_hillstrom
from .estimate import compare
from .evaluate_uplift import diff_means
from .simulate_bias import nonlinear_propensity, simulate_bias, true_propensity

RESULTS = Path(__file__).resolve().parents[2] / "results"
WEIGHTS = {"log_history": 1.0, "recency": -1.0}
SCENARIOS = {"linear": true_propensity, "nonlinear": nonlinear_propensity}


def _one(rct, scenario, strength, seed, n_boot, threads):
    warnings.filterwarnings("ignore")
    learners.THREADS = threads
    obs = simulate_bias(rct, WEIGHTS, strength, seed=seed, propensity=SCENARIOS[scenario])
    out = compare(obs, n_boot, seed).rename_axis("estimator").reset_index()
    return out.assign(scenario=scenario, seed=seed, n=len(obs))


def run(rct, scenarios=tuple(SCENARIOS), reps=200, strength=1.5, n_boot=100, n_jobs=-1):
    jobs = [(s, seed) for s in scenarios for seed in range(reps)]
    threads = None if n_jobs == 1 else 1
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_one)(rct, s, strength, seed, n_boot, threads) for s, seed in jobs
    )
    return pd.concat(parts, ignore_index=True)


def summarize(runs, truth):
    err = runs.est - truth
    df = runs.assign(
        err=err, sq=err**2, cover=(runs.lo <= truth) & (truth <= runs.hi), width=runs.hi - runs.lo
    )
    g = df.groupby(["scenario", "estimator"], sort=False)
    out = g.agg(
        bias=("err", "mean"),
        sd=("est", "std"),
        mse=("sq", "mean"),
        coverage=("cover", "mean"),
        ci_width=("width", "mean"),
        reps=("seed", "size"),
    )
    out["rmse"] = np.sqrt(out.pop("mse"))
    out["coverage_se"] = np.sqrt(out.coverage * (1 - out.coverage) / out.reps)
    return out[["bias", "sd", "rmse", "coverage", "coverage_se", "ci_width", "reps"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--strength", type=float, default=1.5)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    args = parser.parse_args()
    rct = load_hillstrom()
    truth = diff_means(rct.y, rct.t)[0]
    runs = run(rct, args.scenarios, args.reps, args.strength, args.n_boot, args.jobs)
    RESULTS.mkdir(exist_ok=True)
    runs.to_csv(RESULTS / "montecarlo_runs.csv", index=False)
    table = summarize(runs, truth)
    table.to_csv(RESULTS / "montecarlo_summary.csv")
    print(f"truth {truth:.4f}")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
