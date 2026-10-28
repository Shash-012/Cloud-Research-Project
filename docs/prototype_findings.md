# Prototype findings (Phase 2, synthetic environment)

Every number below is read from `results/tables/*.csv`, which the scripts in `scripts/p2_*.py` produce.
The sweep covers 14 policies × 4 workloads × 10 evaluation seeds (1 to 10): 560 runs, 0 failed,
2 h 15 min on 6 worker processes. Metrics exclude the 200 warm-up steps.

## Acceptance check 2

| # | Check | Outcome | Pass |
|---|---|---|---|
| 1 | `proposed` violation rate below `opt` on at least 3 of 4 workloads | 4 of 4. `proposed`: 0.0000 / 0.0000 / 0.0000 / 0.0011 (daily_ramp_burst / flash_crowd / sawtooth / step); `opt`: 0.5017 / 0.5011 / 0.4964 / 0.5031 | yes, but see "OPT oscillates" below |
| 2 | `abl_min_cost` picks the smallest feasible pool in every non-fallback decision | 57,502 of 57,502 decisions (40 runs) | yes |
| 3 | End-to-end coverage of `full` in [0.93, 0.97] | 0.9700 (3880 of 4000 test rows) with z = 1.6449. This is exactly on the upper edge. The conformal z_cal = 1.2256 would give 0.9480 | yes, at the band edge |
| 4 | Power-model sensitivity of the (cost, energy) front of feasible candidates, recomputed from the `proposed` decision logs (49,591 decisions) | square, ΔP/P_idle = 4: 99.6% of decisions have more than one front point (mean 6.5 points). Ratio 1: 100% have exactly one. Ratio 1.42: 0.4% multi-point. Ratio 2: 64.9%. Linear at ratio 4: 0% (energy rises with n). Cubic at ratio 4: 99.6% | yes |
| 5 | All tests pass; same seed gives identical logs | `pytest -q`: 105 passed. Two fresh `proposed`/flash_crowd/seed-3 runs give byte-identical `run_log.parquet` and `decisions.jsonl.gz`, and the run log is identical to the sweep's copy | yes |

For item 4 the script counts "many decisions" as at least 20% and "almost always" as at least 95%. The roadmap gives no numeric thresholds; both observed values are far from these cut-offs. The jump between ratio 1.42 (0.4%) and ratio 2 (64.9%) reproduces the analytic threshold 1/u_slo² ≈ 1.42.

## Synthetic sweep: violation rate, cost, energy (mean over 10 seeds)

Cost is USD per simulated day and energy is kWh per day. Order of workloads: daily_ramp_burst / flash_crowd / sawtooth / step.

| Policy | Violation rate | Cost (USD) | Energy (kWh) |
|---|---|---|---|
| cpu_threshold | 0.0010 / 0.0406 / 0.0014 / 0.0252 | 34.12 / 24.91 / 32.08 / 33.57 | 65.73 / 39.44 / 69.10 / 72.19 |
| rt_threshold | 0.0027 / 0.0870 / 0.0055 / 0.0393 | 28.13 / 16.84 / 29.93 / 30.37 | 67.32 / 40.47 / 69.83 / 73.78 |
| opt | 0.5017 / 0.5011 / 0.4964 / 0.5031 | 23.88 / 15.58 / 23.49 / 26.54 | 65.78 / 35.81 / 66.23 / 74.41 |
| **proposed** | 0.0000 / 0.0000 / 0.0000 / 0.0011 | 34.07 / 22.65 / 34.94 / 36.87 | 64.29 / 37.93 / 65.96 / 70.92 |
| abl_stage1_linear | 0.0801 / 0.0855 / 0.0802 / 0.0827 | 31.92 / 21.04 / 32.65 / 34.44 | 66.33 / 39.42 / 67.95 / 73.12 |
| abl_single_member | 0.0000 / 0.0000 / 0.0000 / 0.0010 | 34.11 / 22.51 / 34.95 / 36.86 | 64.28 / 37.95 / 65.95 / 70.91 |
| abl_no_uncertainty | 0.0000 / 0.0001 / 0.0000 / 0.0015 | 31.02 / 20.34 / 31.65 / 33.36 | 65.48 / 38.33 / 67.26 / 72.41 |
| abl_epistemic_only | 0.0000 / 0.0000 / 0.0000 / 0.0015 | 31.30 / 20.61 / 31.87 / 33.55 | 65.35 / 38.27 / 67.15 / 72.32 |
| abl_no_propagation | 0.0000 / 0.0000 / 0.0000 / 0.0014 | 31.88 / 21.27 / 32.47 / 34.21 | 65.08 / 38.13 / 66.89 / 72.01 |
| abl_quantile_nn | 0.0000 / 0.0000 / 0.0000 / 0.0015 | 31.69 / 21.01 / 32.49 / 34.28 | 65.16 / 38.17 / 66.93 / 71.92 |
| abl_quantile_hgb | 0.0000 / 0.0000 / 0.0000 / 0.0015 | 31.78 / 21.05 / 32.42 / 34.22 | 65.11 / 38.16 / 66.87 / 71.97 |
| abl_min_cost | 0.0219 / 0.0437 / 0.0294 / 0.0262 | 27.26 / 16.85 / 27.93 / 29.99 | 68.08 / 40.06 / 69.56 / 74.69 |
| abl_two_objective | 0.0000 / 0.0010 / 0.0002 / 0.0019 | 31.89 / 19.22 / 32.51 / 34.70 | 65.08 / 38.58 / 66.57 / 71.56 |
| abl_with_rate | 0.0000 / 0.0000 / 0.0000 / 0.0013 | 34.17 / 22.79 / 35.06 / 36.97 | 64.27 / 37.93 / 65.93 / 70.89 |

Confidence intervals, p95 response times, churn, fallback rate and realised coverage are in
`results/tables/synthetic_main_results.csv`. Wilcoxon tests of `proposed` against every other policy (with Holm correction) are in `synthetic_paired_tests.csv`. The bar chart is `results/figures/synthetic_policy_bars.png`.

## Model quality

* **Stage 1 (`stage1_r2.csv`).** The t-test drops `net_in_kbps` (p = 0.27): the rate does not change with the action. The R² filter drops `queue_len`: its best candidate is HGB with R² = 0.856, below 0.90, so it is rejected outright. The selected metrics are cpu_util, mem_util and disk_iops.
    * The chosen models are `mlp_ensemble` for cpu_util (R² 0.968 vs 0.919 for basis_linear) and mem_util (0.937 vs 0.885), and `paper_linear` for disk_iops (R² 1.0000).
    * On the two metrics that are exactly linear in the features (net_in_kbps, disk_iops), the best non-linear candidate trails the best linear one by 0.00007 and 0.00010 R². I count that as "matching" (fourth-decimal CV noise). On the other three metrics the non-linear candidates are 0.05 to 0.17 R² better.
* **Stage 2 (`stage2_accuracy.csv`, true after-metrics as inputs).**
    * The ensemble-mean R² on log RT is 0.9951, and MAE is 6.58 ms on the 2596 test rows below 500 ms.
    * Retraining with the same seeds reproduces the predictions bit for bit.
    * The best hyperparameters are (32, 16), lr 3.4e-4, dropout 0, batch 64. The best trial ran all 300 epochs without early-stopping, so a larger `max_epochs` might help slightly.
* **Quantile models (`quantile_coverage.csv`, true inputs).** Coverage at τ = 0.95 is 0.9313 (QuantileNet) and 0.9300 (QuantileHGB). Both lie inside [0.93, 0.97], but HGB sits exactly on the lower edge.

## Calibration end to end (`calibration.csv`, coverage at the 0.95 level, test split)

| Variant | Coverage | Coverage in SLO bin | Mean σ (ale / epi / stage 1) | z_cal | Coverage with z_cal |
|---|---|---|---|---|---|
| none | 0.4715 | 0.361 | 0 | – | – |
| epistemic | 0.5840 | 0.488 | 0.031 (0 / 0.031 / 0) | 17.72 | 0.956 |
| ale_epi | 0.9073 | 0.736 | 0.110 (0.101 / 0.031 / 0) | 2.645 | 0.954 |
| **full** | **0.9700** | 0.973 | 0.308 (0.115 / 0.060 / 0.270) | 1.226 | 0.948 |
| full, one member | 0.9685 | 0.975 | 0.305 | 1.254 | 0.950 |
| full, paper_linear stage 1 | 0.9270 | 0.889 | 0.583 | 2.384 | 0.952 |
| quantile_nn | 0.8735 | 0.909 | – | – | – |
| quantile_hgb | 0.8695 | 0.865 | – | – | – |

`epistemic` alone is far below nominal, which reproduces the earlier prototype finding. Propagating stage-1 error is what brings coverage up to nominal: stage-1 uncertainty is the largest part of σ. The quantile models reach 0.93 with true inputs but only 0.87 end to end, because they ignore stage-1 error.

## What did not behave as expected

1. **OPT oscillates, which is why it violates the SLO half the time.** The base paper's `paper_linear` cpu estimator learned weights 0.892 and 0.846 for [m·w/n, m·k/n]. Since the two features sum to m, it predicts roughly 0.87 × the current CPU whatever the action.
    * On test rows where the pool is halved from 45–70% CPU, it predicts 64% against a true 98.5%. The chosen MLP predicts 98.8%.
    * Scanning from k_min upward, OPT therefore picks k = −7 at a healthy pool. At the next step the pool is overloaded, every candidate looks infeasible, and OPT falls back to k_max = +7. This 2-step cycle repeats: violation ≈ 0.50, fallback ≈ 0.48, churn ≈ 1238 actions per day.
    * This follows from the baseline exactly as specified (Step 2.11), not from a code bug. Item 1 passes by a wide margin because of it. A fairer OPT comparison will need the real base-paper data or the CloudSim Plus environment.
2. **`full` coverage sits exactly on the 0.97 edge, and conditional coverage is uneven.** Coverage is 0.97 in the bin containing the SLO but 0.885 in the 271–6051 ms predicted-RT bin (the overload region) and above 0.99 elsewhere.
    * The cause is a single global Gaussian covariance for stage-1 error. The residuals are strongly heteroscedastic: CPU saturated at 99.5% before scaling carries no information about the post-scaling load.
    * As the roadmap instructs, the controller keeps z = 1.6449 because coverage is inside the band. Setting `slo.use_conformal: true` would use z_cal = 1.226 (coverage 0.948).
3. **`proposed` is conservative.** It has near-zero violations, and realised coverage of its chosen bound is 0.94–0.97 across workloads. But its cost is about the same as `cpu_threshold` and 2–4 USD per day above the S=1 ablations. `abl_min_cost` saves about 20% cost at a 2–4% violation rate.
4. **High churn.** `proposed` acts in about 60% of intervals, almost all of them ±1. Monte Carlo noise and nearly tied Tchebycheff scores make it flip between neighbouring pools, and the roadmap specifies no hysteresis or cooldown for it. The lag between load and pool is still 1 step (correlation 0.889). The one-episode check took 43.8 s, under the 1-minute limit.
5. **Ablations differ little in violations on this synthetic system.** Every S=1 mode keeps violations below 0.2%, because the system's noise is small relative to the gap between neighbouring pool sizes. They differ in realised coverage (0.45–0.79 vs 0.94–0.97 for `full`) and in cost. Swapping in the linear stage 1 (`abl_stage1_linear`) is the only ablation that clearly raises violations (about 8%).
6. **`abl_with_rate` and `abl_single_member` are nearly identical to `proposed`.** Adding the request rate gives no gain (test R² 0.9947 vs 0.9951). One member loses little because σ is dominated by the stage-1 term.

## Deviations from ROADMAP.md

1. **ROADMAP.md was at the project root.** It was moved to `docs/ROADMAP.md` (with the PDF copy) to match the Step 0.2 layout.
2. **`calibration.seed` is 1001, not 7.** Seed 7 is also an evaluation seed, which would break the rule that calibration and evaluation seeds never overlap.
3. **Baseline thresholds are still the roadmap's placeholders.** The base paper's values were not available.
4. **`.gitignore` uses `/models/` instead of `models/`.** The unanchored pattern also ignored `src/autoscaler/models/`, so the stage-1 code was briefly left untracked. A follow-up commit fixed the pattern and added that code.
5. **The propagation spread uses the population divisor 1/(MS) instead of 1/(MS−1).** With 1/(MS−1), S=1 modes divide by M−1 but `full` divides by MS−1. That breaks two of the roadmap's own required tests: Σe = 0 ⇒ `full` = `ale_epi`, and σ non-decreasing in Σe. With 1/(MS), the spread equals σ_epi² + σ_stage1² exactly. At M = 5, S = 32 the difference is 0.6%; in S=1 modes it removes the M/(M−1) Bessel factor. When Σe is all zeros, `full` skips the identical draws.
6. **Constants added to `configs/prototype.yaml`.** Rule 3 asks for every constant in config, so these were added: `time.decision_every`, `system.u_max`, `system.rt_floor_ms`, `system.complexity.floor`, `telemetry.*`, `workloads.*`, `slo.z`, `slo.use_conformal`, and the seeds and tie tolerance in `stage1`, `ensemble`, `quantile` and `calibration_report`. Other new sections are `tuning` (search ranges), `quantile`, `propagation.resample` and `calibration_report`. `experiments.yaml` gained `policies`, `ablations` (the Phase 4 list) and `sweep.workers`.
7. **Stage-1 features use the effective action k = n_after − n_before**, that is, after clamping to [1, 30]. The controller scores only in-range actions, where the effective action equals k.
8. **`rt_p95_ms` in the synthetic run log is mean × ln 20.** The formula system has no per-request times, so exponential sojourn times are assumed.
9. **The Phase 5 analysis that the sweep calls is a minimal subset.** `eval/metrics.py` (`summarise`), `eval/stats.py` (t intervals, Wilcoxon, Holm) and `scripts/p5_01_aggregate.py` are included. Hypervolume, the report builder and the remaining figures belong to Phase 5.
10. **Two scripts were added.** `scripts/p2_07_acceptance.py` evaluates Acceptance check 2. Quantile training (Step 2.7) runs inside `scripts/p2_03_train_rt.py`.
11. **A `MeanNet` class (same body, one output) holds the MSE-trained OPT network**, so it shares the ensemble's save and load code.
