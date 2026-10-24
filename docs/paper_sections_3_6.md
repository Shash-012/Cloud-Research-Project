# Paper reading notes: Sections 3–6

These notes record the implementation consequences of the second pass through
Sections 3–6 of the base paper. They are a traceability aid, not a substitute
for the paper.

## Section 3 — system and workload model

- Keep the SimPy environment behind the environment boundary so a later
  CloudSim Plus adapter can replace the simulator.
- Treat application profiles, workload shapes, monitoring interval, VM limits,
  and boot delay as configuration inputs.
- Use paired workload seeds when comparing policies.

## Section 4 — metric selection and response-time modelling

- Keep perturbation, metric generation, dynamicity filtering, and response-time
  model fitting as separate stages.
- The configured `alpha: 0.001` and `r2_threshold: 0.9` are the primary
  reproduction choices; sensitivity values are recorded in
  [`decisions.md`](./decisions.md).
- Synthetic metrics validate the selection procedure only, not production
  `collectl` measurements.

## Section 5 — autoscaling policies

- Preserve RT-threshold, CPU-threshold, reactive OPT, and proactive OPT as
  separate policy surfaces.
- Apply the same VM pool bounds and boot delay to every policy comparison.
- Proactive candidate search uses `k_min..k_max` and skips invalid capacity
  candidates through `candidate_ks`.

## Section 6 — evaluation and reporting

- Use paired seeds and report both overall response time and violating-request
  response time.
- Report VM use as the total attached-VM interval usage and its percentage of
  the configured maximum.
- Run the full protocol before drawing conclusions from short smoke runs, and
  retain the zero-boot-delay ablation.

## P0 traceability

The configuration loader is implemented in
[`src/autoscale/config.py`](../src/autoscale/config.py), with its default
settings in [`configs/default.yaml`](../configs/default.yaml) and focused
coverage in [`tests/test_config.py`](../tests/test_config.py).
