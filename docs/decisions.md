# Decisions log

The base paper contains inconsistencies and leaves several settings unstated. Each choice below is deliberate and
should be repeated in the methodology section of the report.

| Ambiguity | What the paper says | Our choice (config key) |
|---|---|---|
| t-test significance | p < 0.001 (Sec 4.1) vs alpha = 0.05 (Sec 5) | `alpha: 0.001`; sensitivity check at 0.05. No multiple-comparison correction (as in paper) |
| R2 threshold | 0.9 (method) vs 0.95 (conclusion) | `r2_threshold: 0.9`; sensitivity run at 0.95 |
| Scale-in trigger (Eq. 10) | text: below 60% of QoS; formula evaluates to 40% | `reactive_lower_frac: 0.6` (text value) |
| Algorithm 1 loop | printed as [Kmin, Kmin]; no w+k guard | `[Kmin, Kmax]`; candidates with w+k < 1 or above the pool cap are skipped (`candidate_ks`) |
| "One less resource" prose vs code | prose says fewer VMs, code moves to next larger k | code (scan upward, first safe k) |
| Early stopping | 100/500 (Sec 4.5) vs 50/100 (Sec 6.3) | final fit 100/500; HPO trials 50/100 |
| Input normalisation | none (Sec 4.5) vs searchable (Sec 6.3) | searchable option in HPO (`normalization`) |
| Target scaling | outputs "not standardised" | target standardised internally and un-standardised in `predict()`: a pure reparameterisation of the output layer (Adam at lr 0.01 cannot reach ~1000 ms otherwise) |
| t-test variant | not stated | Welch (`equal_var=False`) |
| "OPT baseline" | never defined | same trained models as proactive, reactive trigger (Eq. 10), k by Eq. 8-9. Wajahat's original (1 hidden layer, sigmoid, 8 metrics + RR) is a possible extra ablation level (not implemented) |
| RTavg | "during violations" | both overall mean (`rt_avg_ms`) and mean of violating requests (`rt_avg_viol_ms`) |
| VM used | total with % of max | `vm_used` = sum of attached VMs over all intervals; `% = vm_used / (max_vms * intervals)` |
| Boot delay, pool cap | not stated | 30 s, 14 VMs; set `boot_delay_s: 0` for the ideal-scaling ablation |
| Per-VM -> cluster metrics | not stated | mean across attached VMs |
| Metric prediction range | not stated | Eq. 1 predictions are clipped to the training range +/-10% (prevents e.g. 150% CPU and NN extrapolation) |
| Booting VMs | not modelled in the paper's algorithm | policies see `pending`; OPT converts k (relative to attached w) into a change of the total pool (`k - pending`); threshold policies use a cooldown |
| Training data per observation | not stated | 100 s hold, last 5 settled intervals averaged for before/after; ramps are very slow (4 h period) so load is almost equal across the two windows. Without this the R2 filter rejects every metric because load drift, not scaling, dominates the error |
| NN training rows | not stated | both the "after" and the "before" state of every transition (metrics -> RT measured in that state) |
| HPO scoring | RMSE of f(g(k=0)) | same, on before-state rows of held-out contiguous folds |

## Known limitations (state in the report)

* Metrics are synthetic with a planted ground truth: validates the selection **procedure**, not that specific collectl metrics matter.
* Eq. 1 is linear: it can predict impossible values near saturation (tested in `tests/test_eq1.py`).
* Single synthetic traces stand in for the private webshop trace.
* The simulator has no failures, no network latency and no per-VM heterogeneity.
* The paper's code and data are private, so this is a reconstruction; absolute numbers are not comparable.
