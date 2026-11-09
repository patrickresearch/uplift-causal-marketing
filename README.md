# uplift-causal-marketing

**Did the campaign work, for whom, and whom should we not contact?**
Causal inference and uplift modeling on the Hillstrom e-mail experiment (64k customers) and the Criteo uplift dataset (13.9M users).

![Naive vs. adjusted vs. true effect](docs/figures/hero.png)

**[Rendered notebooks](https://patrickresearch.github.io/uplift-causal-marketing/)** · **Budget calculator:** `make app`

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

**Experiment (Hillstrom, women's e-mail → visit).** ATE +4.5 pp (95 % CI 3.9–5.2) on a 10.6 % baseline.

**Observational version.** Confounding on recency and purchase history (standardized mean difference ≈ 1.0) nearly doubles the naive estimate:

| | Estimate | 95 % CI | Bias |
|---|---|---|---|
| Naive difference | 8.7 pp | 7.8–9.6 | +4.2 |
| PS matching | 5.5 pp | 4.2–7.0 | +1.0 |
| IPW | 5.3 pp | 3.9–6.5 | +0.8 |
| Double ML | 4.8 pp | 3.6–6.0 | +0.3 |

All refuters pass: placebo treatment −0.02 pp, random common cause and data subsets leave the estimate unchanged.

**Uplift.** Normalized Qini on held-out data:

| | Hillstrom | Criteo (2M) |
|---|---|---|
| T-learner | 0.043 (0.015–0.068) | 0.072 (0.060–0.082) |
| X-learner | 0.043 (0.017–0.068) | 0.076 (0.066–0.087) |
| Causal Forest | 0.063 (0.040–0.088) | – |

![Qini](docs/figures/qini.png)

**Targeting.** Contacting only the top 30 % by predicted uplift captures **53 % (43–65 %)** of the incremental visits on Hillstrom and **79 % (71–85 %)** on Criteo, where the effect is far more concentrated (top 10 % → 53 %).

**Segments.** On Hillstrom, 16 % of customers are *Sure Things*: they visit anyway and the e-mail adds nothing measurable (+0.3 pp, CI −2.3 to 2.8). Skipping them saves 16 % of the contact cost at no measurable loss. *Sleeping Dogs* are practically absent (0.02 %): for visits, an e-mail rarely hurts.

![Segments](docs/figures/segments.png)

## Budget calculator

`app/streamlit_app.py` turns held-out uplift scores into a profit curve: set customer base, cost per contact, value per incremental visit and budget; it returns the optimal target share, expected incremental visits and profit with 95 % bootstrap bands. Scores are exported by notebooks 03/04 to `app/scores_*.parquet`, so the app needs no model training and only light dependencies (`app/requirements.txt`).

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
app/                  Streamlit budget calculator + exported scores
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
