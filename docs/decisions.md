# Reproduction decisions

These choices resolve ambiguities in the paper and are kept explicit so that
results can be interpreted as a reproduction of the mechanism, not as a claim
of identical absolute numbers.

| Area | Decision |
| --- | --- |
| Simulator | Seeded Python SimPy discrete-event simulator |
| Monitoring interval | 10 seconds |
| VM pool | Minimum 1 VM, configurable cap, default 14 |
| Scale-out boot delay | 30 seconds; a 0-second ablation is retained |
| Dynamicity test | Welch two-sample t-test, `alpha = 0.001`; sensitivity at `0.05` |
| Accuracy filter | Held-out `R2 > 0.9`; sensitivity at `0.95` |
| Scale-in trigger | `0.6 * QoS`, following the paper's text |
| Proactive search | Scan `Kmin..Kmax`, skip `w + k <= 0`, cap at pool size |
| Early stopping | HPO: patience 50/max 100 epochs; final fit: patience 100/max 500 |
| Network validation | Contiguous blocked three-fold CV; no shuffled folds |
| Evaluation | Ten seeds per cell, paired workload seeds across policies |
| Reporting | Overall mean RT and mean RT for violating requests; total VM use and percentage of maximum |

Synthetic metrics validate the selection procedure only. They must not be
presented as measurements of collectl or as evidence that a particular
production metric is universally causal.
