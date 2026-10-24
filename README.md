# Autoscaling project, Part 1: reproduction of Pintye et al. (2024)

Re-implementation of the ML-based autoscaler of *Enhancing Machine Learning-Based Autoscaling for Cloud Resource
Orchestration* (J. Grid Computing 22:68) in a SimPy cloud simulator, with the RT-threshold and CPU-threshold baselines.

## Install
```bash
# macOS/Linux
python -m venv .venv && source .venv/bin/activate
# Windows PowerShell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
# Windows Command Prompt
python -m venv .venv && .venv\Scripts\activate.bat
pip install -e .
pytest -q                                              # 21 tests, ~10 s
```

## Run (each step writes to results/)
| Step | Command | Produces |
|---|---|---|
| E0 simulator validation | `python scripts/00_validate_sim.py` | M/M/c + Little's law table |
| Perturbation data | `python scripts/01_collect.py` | `results/raw/dataset_<app>.npz` |
| E1/E2 metric selection | `python scripts/02_select_metrics.py` | selected metrics, box plots, PCA, R2 bars |
| Train models + HPO | `python scripts/03_train.py` | `results/models/<app>/` |
| Tune baselines | `python scripts/04_tune_baselines.py` | `results/tables/tuned_baselines.yaml` |
| E3 constant request rate | `python scripts/07_exp_constant_rr.py` | RR/VM/OPT comparison |
| E4-E6 main experiments | `python scripts/05_run_all.py` | `results/raw/runs_<workload>.csv`, time series |
| Figures and tables | `python scripts/06_make_figures.py` | `results/figures`, `results/tables` (md/csv/tex) |

`make all` runs everything. Quick smoke run:
```bash
python scripts/01_collect.py --obs 200 && python scripts/02_select_metrics.py && python scripts/03_train.py --trials 3
python scripts/04_tune_baselines.py --fast && python scripts/05_run_all.py --seeds 2 --duration 1200 && python scripts/06_make_figures.py
```
Full-size run time on one core: data collection ~10 min, HPO ~10-15 min, tuning ~20 min, experiments ~30-60 min.
`05_run_all.py` and tuning use all but one CPU core.

## Layout
`src/autoscale/env` simulator (SimPy), apps, workloads, 128 synthetic metrics | `data` perturbation + dataset |
`selection` t-test and R2 filters | `models` Eq. 1 LR, PyTorch NN, HPO, bundle | `policies` RT, CPU, OPT reactive/proactive |
`runner` experiment runner | `analysis` KPIs, plots, tables | `tests` | `docs/decisions.md` | `cloudsim/` (Part 2).

Configuration is loaded from `configs/default.yaml` and merged with the
application and workload profiles by
[`src/autoscale/config.py`](src/autoscale/config.py). Paper traceability notes
for Sections 3–6 are in
[`docs/paper_sections_3_6.md`](docs/paper_sections_3_6.md).

## Hooks for Part 2
* `env.base.Env`: replace SimPy by a CloudSim Plus bridge.
* `models.metric_lr.MetricLR`: replace Eq. 1 by a non-linear estimator.
* `RTModel.predict_dist(X) -> (mu, sigma)`: sigma is 0 here; an ensemble plugs in. `ModelBundle.estimate_rt` already passes it through.
* `policies/opt_proactive.algorithm1`: swap "first safe k" for a chance constraint / Pareto rule.

## Status of the reproduction (honest summary of the smoke runs)
Verified: simulator matches M/M/c theory and Little's law; metric selection has no false positives on planted distractors, differs
across apps and is stable across 200/400/720 observations; E3 shows OPT (internal metrics) far above RR/VM models.
**Not yet reproduced:** in short smoke runs (2 seeds, 1200 s, coarse tuning grid) OPT-proactive used far fewer VMs than the
CPU rule but did NOT beat it on QoS violations, and was worse than OPT-reactive. Likely causes to investigate (see `docs/decisions.md`):
30 s boot delay with coarse pool granularity, no hysteresis in "smallest safe pool" scale-in, noisy 10 s metrics, and the network
predicting MEAN RT while violations are per-request. Run the full protocol (10 seeds, 2400 s, full tuning grid) before drawing
conclusions, then ablate `boot_delay_s: 0`. Report whatever you find; a non-reproduction is a valid result.
