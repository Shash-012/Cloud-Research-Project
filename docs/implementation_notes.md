# Implementation notes: Phase 3 (CloudSim Plus) and the Phase 4/5 code

CloudSim Plus is pinned to **8.5.7** (`org.cloudsimplus:cloudsimplus:8.5.7`), on JDK 21 (Temurin 21.0.8) and Maven 3.9.16. I checked every engine method used below against the 8.5.7 jar (`javap`) and its sources jar (`cloudsimplus-8.5.7-sources.jar`).

## 1. API used, and substitutions

| Roadmap names | In 8.5.7 | Used as written? |
|---|---|---|
| `new CloudSimPlus(minTime)`, `terminateAt`, `addOnClockTickListener`, `clock()` | present | yes |
| `DatacenterSimple`, `HostSimple`, `PeSimple`, `VmSimple`, `CloudletSchedulerSpaceShared`, `CloudletSimple`, `UtilizationModelFull` | present | yes |
| `DatacenterBrokerSimple.setShutdownWhenIdle(false)` | present | yes |
| `broker.setVmDestructionDelayFunction(null)` | present; `null` installs the default function, which never destroys idle VMs | **adjusted**, see 1.3 |
| `broker.setVmMapper` | present | yes |
| `vm.setStartupDelay`, `vm.isStartingUp()` | present (`ExecDelayable`) | yes |
| `broker.requestIdleVmDestruction(vm)` | present | yes (see 1.3) |
| `cloudlet.addOnFinishListener`, `getStartTime`, `getFinishTime` | present | yes |
| `Log.setLevel(...)` | present (`org.cloudsimplus.util.Log`) | yes, `Level.WARN` |
| `PowerModelHostSimple(maxPower, staticPower)` | present | yes (energy cross-check only) |

### 1.1 Mapping at arrival, not at submission (Step 3.4)
In 8.5.7 the broker calls the VM mapper while processing the `CLOUDLET_CREATION` event that follows `submitCloudletList`, so a cloudlet submitted with a submission delay is mapped at **submission** time. The roadmap's fallback is one-second batches, which would still map up to one second early.

Instead, `ArrivalGenerator` is a `CloudSimEntity` that schedules one `ARRIVAL` event per request and submits that request (delay 0) when the event fires. Mapping therefore happens exactly at the planned arrival time: `ArrivalTimingTest` shows a maximum error of 0 s over 9141 requests.

### 1.2 Processing-update timing (`AccurateDatacenter extends DatacenterSimple`)
Three behaviours of 8.5.7 biased requests lasting tens of milliseconds:

1. **Processed work is truncated to whole units at every update.** `CloudletSchedulerAbstract` computes `(long)(mips * timeSpan)`. At 1000 MIPS one unit is 1 ms, and every event triggers an update of every host, so requests ran up to 14% long.
   *Fix:* `engine.mips_scale: 1000`. Host and VM MIPS and request lengths are multiplied by the same factor, so service times are unchanged and the truncation shrinks to 1 µs.
2. **The next update is never sooner than `minTimeBetweenEvents + 0.01 s`** (hard-coded in `DatacenterSimple.updateHostsProcessing`). A request with under 11 ms of work left finished up to 11 ms late.
   *Fix:* the subclass overrides that method and uses `minTimeBetweenEvents` alone.
3. **`updateCloudletProcessing` silently skips any update within `minTimeBetweenEvents` of the last one, without rescheduling.** Once the 10 ms padding is removed, the update scheduled for a request's finish can be skipped this way. Arrivals to a busy VM trigger no update, so the VM stalled: completions dropped to 0 after about 4 intervals.
   The same guard was also absorbing an endless chain of zero-delay events from `VmAbstract.bootProcessing()`. With `schedulingInterval = 0` that method schedules the next update of a booting VM with delay 0, because it tests `schedulingInterval > -1`.
   *Fix:* the override processes every update except an exact repeat at the same instant. It schedules a new update only if it is earlier than the one already pending, and never with a delay below `minTimeBetweenEvents`.

### 1.3 Scale-in by draining
`requestIdleVmDestruction(vm)` shuts a VM down only if the destruction-delay function marks it destroyable. The broker also calls `requestIdleVmDestruction` after **every** cloudlet return. A function that returned 0 for every draining VM destroyed a VM with 22 queued requests the moment its execution list was briefly empty.

The function therefore returns the default, "never destroy" (the same as passing `null`), except while `ScalingActuator` makes its own request for a drained VM with zero outstanding requests. The request counts only once `vm.getShutdownBeginTime() >= 0`; otherwise it is retried on the next clock tick. `ScalingTest` checks that the drained VM finished all 22 queued requests before it was destroyed, and that no request is lost.

### 1.4 Memory
The broker (`cloudletSubmittedList`, `cloudletCreatedList`, `cloudletFinishedList`) and every scheduler (`cloudletFinishedList`, `cloudletReturnedList`) only ever append. Over 4.8 million requests this exhausts memory. In the 8.5.7 sources none of these lists is read by simulation logic, so `Cluster.releaseFinishedCloudlets()` clears finished entries at every interval boundary. Private fields are cleared through reflection. A 360-step run uses about 240 MB.

### 1.5 Engine resolution
The roadmap asks for `CloudSimPlus(0.001)`, which resolves at least 1 ms. With 1 ms, the M/M/1 check (λ = 15, μ = 25) gave **105.9 to 107.1 ms on five seeds: +6.3%, failing the 5% band**. Requests finished one event-resolution step late about half the time. With **0.1 ms** the same check gives 100.07 to 101.14 ms (mean 100.48). The roadmap says this queueing test decides whether the resolution is fine enough, so `engine.min_time_between_events: 0.0001`.

## 2. Other implementation choices
* **Telemetry and energy** (Steps 3.7, 3.8) are integrated on every clock tick. A clock-tick listener fires when the clock advances, before the new time's events run, and no event happens between ticks, so the state seen at a tick holds for the whole elapsed slice and the integral is exact. This is equivalent to splitting each cloudlet's busy period across interval boundaries. A marker event at every boundary guarantees a tick there.
* **Energy per VM in an interval** is `alive_s * P(busy_s / alive_s) / 3600` Wh, with 0 utilisation while booting. **Cost** uses time-weighted VM-seconds (running + booting + draining), so a VM is billed until it is destroyed.
* **Pool limits:** running + booting stays within [vm_min, vm_max], and running + booting + draining never exceeds the 30 hosts (one VM per host).
* **The bridge** is the protocol of Step 3.11, plus `n_draining` and `rate` in `observe` and `fallback` in `action`. Missing values are sent as `null`. Both sides time out (`bridge.timeout_s`): Java exits with code 3 and a clear message, and Python logs the traceback to `bridge_error.txt`.
* **`--remote`** serves even a Java baseline from Python. The Java and Python threshold policies produce byte-identical run logs for the same seed (`tests/test_bridge.py`).
* **Python metrics for CloudSim Plus** are `cpu_util, mem_util, net_in_kbps, queue_len, throughput_rps` (Step 3.7; there is no disk metric). `schema.metrics_for(source)` selects the list.
* **The Phase 2 modules were changed only where Phase 3 needs it,** and their 105 tests still pass:
  * `--source cloudsim` for `p2_02`, `p2_03` and `p2_04`;
  * `GroupKFold` by `run_id` in `MetricEstimator.fit(group_col=...)`;
  * grouped splits in `split_from_config`;
  * table names with a `_cloudsim` suffix;
  * `merge_cloudsim()`, which supplies an effective SLO of rt_ms / s;
  * `summarise()` counting draining VMs, with a lag of 2 intervals in CloudSim Plus.
* **The counterfactual Hold runs** (Step 3.16) set a pool for each pair, since HoldPolicy never changes the pool. The pool is the larger of the pool giving u0 ~ U(0.5, 0.9) at STEP and the pool keeping u <= 0.95 up to STEP+2. Each run stops at STEP+3, and all runs of a pair are identical before STEP (this is checked).
* **The speed benchmark** (Step 3.13 item 7) is a Hold run with 20 VMs instead of the default 4. With 4 VMs the queue on `daily_ramp_burst` grows without bound from the first interval. A `cpu_threshold` run is timed as well.

## 3. capacity_scale, run time, full-grid estimate
**capacity_scale: s = 1.0.** Step 3.13 item 7: one 360-step run on `daily_ramp_burst` (4.85 M requests) took **7.7 min** (Hold, 20 VMs) and **7.3 min** (`cpu_threshold`), with two runs side by side (`results/tables/cloudsim_speed.csv`). That is under the "about ten minutes" limit, so no scaling was needed. The validation tests run at s = 1 and at s = 0.25, the documented fallback; both pass (M/M/1 at 0.25: 398.66 ms vs 400 ms).

**Measured run times.**

| Runs | Concurrent processes | Wall clock per run | Requests per run |
|---|---|---|---|
| Hold / `cpu_threshold`, `daily_ramp_burst`, 360 steps | 2 | 7.3 to 7.7 min | 4.85 M |
| RandomTarget calibration, `random_walk` | 4 | mean 7.8 min (3.9 to 15.1) | 2.63 M |
| RandomTarget calibration, `daily_ramp_burst` (visits overload) | 4 | mean 20.2 min (15.4 to 27.8) | 4.86 M |
| `proposed` through the bridge, `step`, 360 steps | 11 (counterfactual running alongside) | 37.2 min | 5.24 M |
| Counterfactual Hold runs (43 to 123 steps) | 10 | 350 runs in 100 min | |

The engine processes about 10.8 k requests/s per run when two runs share the machine (16 logical cores, 16 GB). Total throughput levels off at about 16 to 22 k requests/s whether 2, 4 or 10 runs are in flight. Long queues, as in RandomTarget's overload visits or under-provisioned baselines, slow a run further. A Python policy adds about 40 ms per decision, roughly 15 s per run, which is negligible.

**Full-grid estimate.** Requests per run (mean over seeds 1 to 10): `daily_ramp_burst` 4.86 M, `step` 5.23 M, `flash_crowd` 3.01 M, `sawtooth` 5.13 M.

* Main grid: 400 runs, 1.82 G requests.
* Ablations: 200 runs, 0.79 G requests.
* Sensitivity: 120 runs, 0.58 G requests.
* **Total: 720 runs, about 3.2 G requests.**

| Workers | Basis | Estimated wall clock |
|---|---|---|
| 1 | 10.8 k req/s | about 82 h |
| 2 | 21.6 k req/s total | about 41 h |
| 4 | 16 to 22 k req/s total (plateau) | about 40 to 55 h |
| 8 | same plateau, so no gain over 4 | about 40 to 55 h |

The upper end allows for runs slowed by long queues (`opt` oscillating, under-provisioned `rt_threshold`). With **s = 0.25** (Step 3.1) the request count, and so the time, falls by about 4x to roughly 10 to 14 h with 4 workers, at the cost of re-validating at that scale (the tests already cover s = 0.25). Memory is about 250 MB per simulator, so 4 to 8 workers fit in 16 GB alongside the Python bridges.


## 4. Phase 3 results and acceptance checks

**Validation** (`mvn test`, 12 tests; Step 3.13). The rule-3 tests all passed before any data was collected.

| Check | Result |
|---|---|
| Arrival timing | max \|mapping clock - planned arrival\| = 0 s over 9141 requests |
| Determinism | byte-identical `run_log.csv` for the same seed; a different seed differs |
| M/M/1 (λ = 15, μ = 25, 324 k requests) | **100.07 ms vs 100 ms (+0.07%)**; s = 0.25: 398.66 vs 400 ms (-0.34%) |
| Utilisation (λ, n = 150, 8 / 60, 4 / 300, 20) | 75.25 / 60.07 / 60.14% vs 75 / 60 / 60%; s = 0.25: 75.14 / 59.44 / 59.95% |
| Start-up | VM requested at 60 s serves its first request at 90.03 s (delay 30 s) |
| Scaling (+3 then -2, no load) | 7 then 5 VMs; a drained VM finished its 22 queued requests; no request lost |
| Energy cross-check (linear) | EnergyMeter 606.5279 Wh vs CloudSim Plus host power model 606.5269 Wh (**0.0002%**) |
| Java vs Python thresholds | byte-identical run logs for `cpu_threshold` and `rt_threshold` (`tests/test_bridge.py`) |
| Bridge timeout | Java exits with code 3 ("bridge: no answer from Python within the timeout") |

**Calibration data** (Step 3.14). 20 RandomTarget runs gave **3,958 transitions**: every k from -7 to 7, no missing values, 8.1% above the SLO and 5.6% with more than 15% rate drift. That is far fewer than the 20,000 synthetic rows, because each run gives about one transition per two intervals.

**Retraining** (Step 3.15, grouped by run_id, `models/cloudsim/`).
* Stage 1 selected cpu_util (MLP ensemble, CV R² 0.904), net_in_kbps (0.978) and throughput_rps (0.972).
* mem_util (0.80) and queue_len (0.11) failed the R² filter.
* Stage-2 training uses `ensemble.max_epochs: 2000` (see section 4.1). Ensemble R² on log RT: 0.908 (true inputs).
* **End-to-end coverage of `full`: 0.9675** (z = 1.6449), inside [0.93, 0.97].
* Coverage in the SLO-containing bin: 0.931. `epistemic` alone: 0.512. Conformal z_cal = 1.544 would give 0.965.

**Counterfactual** (Step 3.16, 50 pairs x k = -3..3 = 350 runs, `results/tables/counterfactual_by_k.csv`).

| k | -3 | -2 | -1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| coverage of U_k | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| median abs error (ms) | 48.1 | 7.7 | 5.2 | 3.3 | 3.7 | 2.3 | 1.8 |

**Acceptance check 3 is not met.** The roadmap requires 0.90 to 0.98 for every k; k = -2..+3 over-cover. The other two items pass:
* every simulator test passes;
* a full 360-step `proposed` run through the bridge has 360 decisions, each with every valid candidate, and the k applied in Java matches the Python decision at every step.

### 4.1 Attempted fix and why the check stays unmet

**Diagnosis.**
* For k >= -1 the outcome is nearly deterministic: median measured RT 40 to 48 ms, log spread about 0.03, none above the SLO. The model also overestimates it by about 5%.
* On low-load test rows (RT < 60 ms) the actual log residual spread is 0.023, but the ensemble's aleatoric σ was 0.17 (v1 models).
* Any bound of reasonable width therefore covers all 50 pairs. To score at most 0.98, at least 1 of the 50 must fall outside the bound, which needs millisecond precision around 40 ms.

**What was tried.** Each candidate was evaluated on the calibration and test splits and on the existing 350 counterfactual runs. The simulations do not depend on the model, so no re-simulation was needed.

| Change | End-to-end coverage | SLO-bin coverage | Counterfactual coverage, k = -3 / -2..+3 |
|---|---|---|---|
| v1 (max_epochs 300; all members stopped at the cap) | 0.951 | 0.856 | 0.94 / 1.00 |
| **v2: max_epochs 2000, so early stopping governs (kept)** | 0.964 to 0.968 | 0.900 to 0.931 | 0.96 / 1.00 |
| v2 with dropout 0 (aleatoric σ at low load 0.133, actual 0.037) | 0.951 | 0.856 | 0.80 / 0.98 to 1.00 |
| v1 with group-conditional ("Mondrian") conformal z per predicted-RT bin, 3 to 6 bins | 0.945 to 0.955 | | 0.94 to 1.00 per k |

None passes for every k. The banded conformal results swing with the number of bins, and choosing the bin count that passes on the counterfactual would tune the method on the test, so it was not adopted.

**The band is statistically fragile.** With 50 pairs per k, a perfectly calibrated bound (true coverage 0.95) lands in 45 to 49 of 50 with probability 0.885 per k, so all 7 k pass with probability **0.43** (0.34 at 0.96, 0.40 at 0.94). In this simulator the low-load outcomes are also nearly deterministic, so the observed 1.00 reflects conservative bounds in a region where they cannot change a decision: U is about 75 ms against a 250 ms SLO.

**Kept:** the v2 models (early stopping instead of the 300-epoch cap). They improve coverage in the decision-relevant SLO bin from 0.856 to 0.931 and k = -3 from 0.94 to 0.96. The v1 models are backed up in `results/runs/models_cloudsim_v1/` (ignored by git). Passing the per-k band would need more pairs per k (fewer chance failures) together with a regime-aware error model; that is left for a decision.

## 4.2 Known limitation (accepted 2026-10-29)
The per-k counterfactual coverage band of acceptance check 3 (0.90 to 0.98 for every k) is **not met and is accepted as a documented limitation**, by decision of the project owner. The full grid proceeds with the current (v2) models.

* **What:** U_k covers 50 of 50 counterfactual outcomes for k = -2..+3, while k = -3 (overload) is in band at 0.96.
* **Why it is acceptable:**
  * The over-coverage is conservative, not unsafe.
  * It occurs only at low load, where outcomes are nearly deterministic (about 40 to 48 ms) and U (about 75 ms) is far below the 250 ms SLO, so it cannot change a feasibility decision.
  * Coverage where decisions are made is within target: end-to-end 0.9675, SLO bin 0.931.
  * The band is statistically fragile: a perfectly calibrated bound passes all 7 k with probability 0.43 at 50 pairs per k.
* **Effect on results:** at most slightly cautious scale-in at low load (a little extra cost and energy). No effect on SLO-violation results is expected.
* **How the final report presents it:** as a limitation with these numbers. The per-k results are written to the report from `counterfactual_by_k.csv`.
* **Possible future work:** more pairs per k (150 to 200) and a load-dependent (heteroscedastic) stage-1 error model or regime-conditional conformal calibration chosen on calibration data only.

## 5. Deviations from ROADMAP.md
1. `engine.min_time_between_events` is 0.0001 s, not 0.001 s: 1 ms failed the M/M/1 check by +6%. See 1.5.
2. `engine.mips_scale` is 1000, and `AccurateDatacenter` overrides two `DatacenterSimple` methods: three 8.5.7 timing artefacts. See 1.2.
3. Arrivals are submitted one event per request rather than in one-second batches, which maps them exactly at arrival. See 1.1.
4. The draining VM is destroyed through a destruction function that is non-default only during our own request. See 1.3.
5. Finished cloudlets are pruned from broker and scheduler lists (memory). See 1.4.
6. **Speed benchmark:** the Hold run uses 20 VMs, since 4 would overload without bound; a `cpu_threshold` run is timed as well.
7. **Counterfactual pairs:** STEP is drawn from [40, 120] instead of [40, 357], which bounds compute (350 runs took 100 min). Each pair uses its own Hold pool and the runs stop at STEP+3.
8. **Telemetry and energy** are integrated on clock ticks rather than from each cloudlet's start and finish times. This is exactly equivalent, and the energy cross-check is within 0.0002%.
9. **Cost** uses time-weighted VM-seconds rather than end-of-interval counts. A VM is billed until it is destroyed.
10. **Pool limit:** running + booting + draining is capped at the 30 hosts (one VM per host).
11. **Bridge messages** carry extra fields (`n_draining`, `rate`, `fallback`) and use `null` for missing values.
12. **Metric set and Phase 2 changes:** CloudSim Plus metrics are `throughput_rps` instead of `disk_iops` (as Step 3.7 defines). The Phase 2 modules gained source-specific options, as listed in section 2.
13. **ReportLab** (>= 4.0) was added to `requirements.txt` for the Step 5.4 report. Pandoc and LaTeX are not installed.
14. **CloudSim stage-2 training uses `max_epochs: 2000`** (an override in `cloudsim.yaml`; the synthetic models keep 300). At 300 every member stopped at the cap, so early stopping never acted; see 4.1.
15. **Smoke grid:** besides the specified main-part smoke (2 policies, 1 workload, 2 seeds, shortened runs) in both environments, a few cheap synthetic runs exercise the ablation and sensitivity code paths (1 ablation, the 12 sensitivity settings for seed 1).
