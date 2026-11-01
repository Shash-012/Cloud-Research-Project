# Failed runs

**Runs that failed twice: none.** Every planned run finished with status `ok`:

| env | part | planned | ok |
|---|---|---|---|
| synthetic | main | 400 | 400 |
| synthetic | ablations | 200 | 200 |
| synthetic | sensitivity | 120 | 120 |
| cloudsim | main | 400 | 400 |
| cloudsim | ablations | 200 | 200 |
| cloudsim | sensitivity | 120 | 120 |

The manifest (`results/runs/manifest.csv`, append-only) also holds 16 CloudSim attempts that failed once and
succeeded on the retry. The analysis keeps the last row per run, so only the successful retry is used. All 16 ended
with `java exit 3` from the Python-Java bridge, for one of two reasons:

1. **Bridge timeout (8 runs).** `configs/cloudsim.yaml` sets `bridge.timeout_s: 120`. For `opt` on `step` and
   `flash_crowd`, some decision intervals took longer than 120 s of wall clock to simulate (a long request queue
   under overload), so the bridge timed out and Java exited with code 3. These runs were retried with
   `results/runs/cloudsim_timeout1800.yaml`, which is identical to `configs/cloudsim.yaml` except
   `bridge.timeout_s: 1800`. The timeout only guards against a hung process. It is not passed to the simulator or
   the policy, so it does not change results.
   - `cloudsim__opt__step__s1`, `cloudsim__opt__step__s4`, `cloudsim__opt__step__s10`
   - `cloudsim__opt__flash_crowd__s1`, `cloudsim__opt__flash_crowd__s4`, `cloudsim__opt__flash_crowd__s7`,
     `cloudsim__opt__flash_crowd__s8`, `cloudsim__opt__flash_crowd__s9`
2. **Host sleep (8 runs, 2026-10-31).** The laptop went into standby during a power cut (11:12-11:46) and again
   (12:31-12:55). The runs in progress lost their bridge connection or timed out. They were re-run with the default
   configuration and finished normally in about 3-4 min each.
   - `cloudsim__abl_quantile_hgb__daily_ramp_burst__s5` to `__s8`
   - `cloudsim__proposed__daily_ramp_burst__s6__pw_square_r1` to `__s9__pw_square_r1`

Also re-run with the 1800 s timeout: the 23 slow `rt_threshold` main runs (step s8-s10, flash_crowd, sawtooth). Under
overload these took up to 17 h each. Four of them were stopped by hand after 30 min on 2026-10-31 to run the `opt`
retries first, then started again from scratch. Stopped attempts never reach the manifest.
