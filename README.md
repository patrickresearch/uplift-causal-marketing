# uplift-causal-marketing

**Did the campaign work, for whom, and whom should we not contact?**
Causal inference and uplift modeling on the Hillstrom e-mail experiment (64k customers) and the Criteo uplift dataset (13.9M users).

![Naive vs. adjusted vs. true effect](docs/figures/hero.png)

## Approach

| Step | Notebook | Method |
|---|---|---|
| ATE in the randomized experiment | [01_experiment](notebooks/01_experiment.ipynb) | Difference in means, Welch CI |
| Confounded observational version | [02_observational](notebooks/02_observational.ipynb) | Covariate-dependent subsampling with known ground truth |
| Correction | 02 | DoWhy backdoor: PS matching, IPW, Double ML (EconML) |
| Robustness | 02 | DoWhy refuters: random common cause, placebo, data subset |
| Heterogeneous effects | [03_uplift](notebooks/03_uplift.ipynb) | T-, X-learner, Causal Forest (LightGBM base models) |
| Uplift evaluation | 03 | Normalized Qini with Poisson-bootstrap CIs, effect-based segments, budget table with CIs |
| Scale | [04_criteo](notebooks/04_criteo.ipynb) | T-, X-learner on 2M users |

### Assumptions

![DAG](docs/figures/dag.png)

The correction in 02 is only valid under no unobserved confounding, overlap and SUTVA.
The biased dataset is built so these hold by construction, which is what makes the true effect known: each RCT unit is kept with probability `e(x)·m/p` (treated) or `(1−e(x))·m/(1−p)` (control), giving `P(kept | x) = m` and `P(T=1 | x, kept) = e(x)`.

Bootstrap CIs for IPW and matching keep the fitted propensity fixed. For IPW this is conservative (Hirano, Imbens & Ridder 2003); for matching the bootstrap is not formally valid (Abadie & Imbens 2008), so DML with its analytic CI is the primary estimate.

Segments are defined on the estimated effect τ̂(x), not on response levels: *Persuadables* τ̂ > δ, *Sleeping Dogs* τ̂ < −δ, otherwise *Sure Things* or *Lost Causes* by baseline response, with δ = ATE/2. Each segment's uplift is re-measured on held-out data.

## Results

![Qini](docs/figures/qini.png)
![Segments](docs/figures/segments.png)

## Setup

```bash
brew install libomp   # LightGBM on macOS
make setup            # uv sync + Chrome for static plotly export
make test
make docs             # executes notebooks, renders HTML to docs/
```

Data is downloaded on first use and cached as parquet in `data/`.

## Structure

```
src/uplift_causal/
  data.py             loaders + parquet cache
  simulate_bias.py    confounded subsample with preserved ATE
  estimate.py         DoWhy estimators + refuters
  learners.py         meta-learners, causal forest
  evaluate_uplift.py  Qini/uplift curves, segments, budget table
  plots.py
notebooks/            thin: call src, show results
docs/                 rendered notebooks + figures
```

## References

- Hillstrom, K. (2008). *The MineThatData E-Mail Analytics and Data Mining Challenge.*
- Diemert, E. et al. (2018). *A Large Scale Benchmark for Uplift Modeling.* AdKDD. Data: [criteo/criteo-uplift](https://huggingface.co/datasets/criteo/criteo-uplift), CC BY-NC-SA 4.0.
- Künzel, S. et al. (2019). *Metalearners for estimating heterogeneous treatment effects using machine learning.* PNAS.
- Chernozhukov, V. et al. (2018). *Double/debiased machine learning for treatment and structural parameters.* Econometrics Journal.
- Wager, S. & Athey, S. (2018). *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests.* JASA.
- Radcliffe, N. (2007). *Using control groups to target on predicted lift.* DMA Analytics Annual.

## License

MIT
