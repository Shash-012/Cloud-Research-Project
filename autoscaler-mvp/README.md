# Pareto-aware, uncertainty-calibrated ML autoscaling for latency SLOs

Prototype extending Pintye, Kovács & Lovas (2024), *Enhancing Machine
Learning-Based Autoscaling for Cloud Resource Orchestration*, J Grid Computing
22:68, https://doi.org/10.1007/s10723-024-09783-1

## Scope: read this first

**This is a synthetic-environment prototype, not a result.** There is no
CloudSim Plus in this repository. The "cloud" is a hand-written queueing model
in `environment.py`, the workload is generated, and the cost and energy models
are closed-form stand-ins with invented constants. Nothing here is calibrated
against a real deployment or against a published trace.

What it does prove: the full pipeline runs end to end, and the two extensions
over the base paper (ensemble uncertainty in the feasibility test, Pareto
filtering over cost and energy) are implemented, testable, and produce sane,
explainable behaviour.

What it does **not** contain, and must not be presented as containing:

- CloudSim Plus integration — the next milestone
- Baseline comparisons — no RT-threshold, CPU-threshold, or reproduction of the
  base paper's own OPT system to compare against. Every number below is
  unopposed.
- Ablations (with/without uncertainty, Pareto vs single-objective, linear vs
  non-linear predictor)
- Online or incremental retraining
- Any oscillation or churn penalty
- More than one application and one workload shape

## Running it

```
pip install -r requirements.txt
python run_all.py            # calibrate -> train -> simulate
python -m pytest tests -q    # decision-layer unit tests
```

Stages are independent and re-runnable:

```
python scripts/01_generate_calibration.py    # -> data/calibration.csv
python scripts/02_train_and_validate.py      # -> models/, figures 01-02
python scripts/03_run_simulation.py          # -> results/, figures 03-05
```

Stage 2 exits non-zero if its success criteria fail, so the control loop never
runs on a predictor that has not been validated.

## How the pipeline fits together

```
workload generator ──> SyntheticApp ──> telemetry (5 metrics + RT)
                                            │
                    calibration run: perturb VM count, log transitions
                                            │
                    ┌───────────────────────┴────────────────────────┐
                    ▼                                                ▼
          MetricEstimator                                  EnsembleRTPredictor
   per-metric LR: (metrics, n, k) -> post-scaling metrics    5 MLPs -> mean, sigma
                    │                                                │
                    └────────────────────┬───────────────────────────┘
                                         ▼
                                  decision.decide
                     1. feasibility:  mean + z*sigma <= RT_SLO
                     2. Pareto filter on (cost, energy)
                     3. tie-break on min cost
                                         │
                                         ▼
                                  apply delta k
```

| Module | Role |
| --- | --- |
| `config.py` | every constant, with the reasoning for the non-obvious ones |
| `environment.py` | workload generator + ground-truth queueing model |
| `calibration.py` | perturbation run producing transition training data |
| `metric_estimator.py` | base paper's Eq. 1–2 layer: post-scaling metric estimates |
| `predictor.py` | MLP ensemble, mean + calibrated total sigma |
| `decision.py` | the contribution: chance constraint + Pareto front |
| `loop.py` | proactive periodic control loop (base paper Algorithm 1) |

## What changed relative to the base paper

**Uncertainty in the feasibility test.** The base paper takes a single network's
point estimate and keeps the smallest \|k\| whose predicted RT clears the QoS
maximum. Here five independently initialised MLPs are trained on bootstrap
resamples, and an action survives only if `mean + 1.65*sigma <= RT_SLO`.

**A Pareto stage instead of first-acceptable-wins.** Among feasible actions,
those dominated on both cost and energy are discarded; the choice is made on
the non-dominated front. When the front has one point this reduces to the base
paper's behaviour, which is the honest baseline case.

**Fallback preserved.** If nothing is feasible, the controller takes the largest
allowed scale-out, matching Algorithm 1 lines 14–15 rather than leaving the
behaviour undefined. This fires 46 times out of 288 decisions in the current
run.

## Results from the current run

Metric selection, using the base paper's R² > 0.9 criterion on the linear
post-scaling estimator:

| metric | CV R² | kept |
| --- | --- | --- |
| cpu_util | 0.965 | yes |
| mem_util | 0.909 | yes |
| net_in_kbps | 0.995 | yes |
| disk_iops | 1.000 | yes |
| queue_len | 0.841 | **dropped** |

Queue length goes as u/(1−u) and no linear model in pool size tracks it near
saturation. The paper's threshold catches exactly this, which is a small piece
of evidence that the criterion earns its place.

Response-time ensemble, held out:

| | |
| --- | --- |
| R², all rows | 0.946 |
| R², rows under 2× SLO | 0.878 |
| MAE, rows under 2× SLO | 9.1 ms |

Uncertainty calibration — the finding worth keeping:

| sigma used in `mean + 1.65σ` | coverage | nominal |
| --- | --- | --- |
| ensemble spread alone | 72.7% | 95% |
| ensemble spread + aleatoric term | 96.3% | 95% |

Ensemble spread on its own is epistemic. In the well-sampled region the five
members agree to within a millisecond or two, so `mean + 1.65σ` sits barely
above the mean, and a controller built on it parks on the SLO boundary while
reporting that it runs at the 95th percentile. Adding an estimated aleatoric
term fixes the coverage, and halves the realised violation rate (39% → 19%) and
cuts p95 latency from 1475 ms to 398 ms in the closed loop. **A single-network
point estimate has no way to express either kind of uncertainty**, which is the
argument for the extension.

Closed loop over 1440 steps, after warm-up:

| | |
| --- | --- |
| median RT | 172 ms |
| p95 RT | 372 ms |
| SLO violation rate | 17.7% |
| mean pool size | 11.8 VMs |
| fallback invoked | 46 / 288 decisions |

## Limitations

**The realised violation rate is 17.7%, not the 5% the chance constraint
implies.** The predictor's held-out coverage is correct, so the gap is in the
control loop, not the model. Two causes, both open: the unobserved complexity
driver is autocorrelated over ~260 steps while the noise calibration assumes iid
residuals, so during a high-complexity phase the constraint is systematically
optimistic for a long stretch; and an action is held for a whole control
interval while conditions drift underneath it. Reporting a chance-constrained
controller without checking realised coverage against nominal would be the easy
mistake here.

**The controller churns.** Mean \|k\| per decision is 1.32 and the VM trace
sawtooths by ±1 continuously. Oscillation penalties were descoped for this
milestone; the trace in `results/figures/04_vm_count.png` shows why they will be
needed.

**The Pareto front only exists because of the power model's shape.** Cost is
linear in pool size. Energy is U-shaped only because per-VM power is
super-linear in utilisation, so total power falls before it rises. That makes
the front's width sensitive to the idle/peak power ratio — at ratios below about
1.42 it collapses to the single cheapest action and the Pareto stage does
nothing. `config.py` documents the arithmetic. This belongs in the write-up as
a sensitivity result, since it bounds when the contribution applies at all.

**The metric estimator is unreliable under saturation.** It assumes metrics
scale as 1/n, which breaks once utilisation clips at 1 and CPU reads 100% at
every pool size. Candidate actions evaluated from a saturated state are
therefore optimistic. In practice the large ensemble sigma in that region marks
them infeasible and the fallback handles it, but it is compensation, not a fix.

**One run, one seed, no error bars.** Every number above comes from a single
seeded run.

## Next milestone

1. Port the environment to CloudSim Plus, keeping the decision layer unchanged —
   `decision.py` depends only on config and numpy, so it should cross intact.
   The Java↔Python bridge is not yet chosen: either ONNX export into a Java ONNX
   Runtime, or a file/socket bridge per control step.
2. Implement the baselines (RT-threshold, CPU-threshold, base paper's OPT) so
   the numbers above have something to be measured against.
3. Ablations: uncertainty on/off, Pareto vs single-objective, linear vs
   non-linear predictor.
4. Replace the closed-form energy stand-in with CloudSim's PowerModel.
5. Ask the base paper's corresponding author (istvan.pintye@sztaki.hun-ren.hu)
   for the dataset and code; the paper says both are available on reasonable
   request.
