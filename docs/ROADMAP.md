---
title: "Implementation Roadmap"
subtitle: "Uncertainty-aware, Pareto-optimal autoscaling: from an empty folder to final results"
author: "Shashwat Tripathi (2024UIN3342)"
date: "October 2026"
---

# How to use this roadmap

This roadmap builds the extension of Pintye, Kovács & Lovas (2024) from nothing, following the original plan:

1. **E1.** Non-linear and ensemble prediction (non-linear metric estimator, deep ensemble for response time).
2. **E2.** Uncertainty quantification: ensemble variance as the main method, quantile regression as an ablation, a chance-constrained decision rule, and Monte Carlo propagation of first-stage error.
3. **E3.** A Pareto-aware decision layer over latency, cost and energy.
4. Evaluation in CloudSim Plus against CPU-threshold, response-time-threshold and the base paper's OPT policy, with ablations, several workloads and repeated seeds.

It ignores the existing `autoscaler-mvp` and `cloudsim-runner` folders. Start in a new folder so nothing from the old code leaks in.

The work runs in six phases. Each phase ends with an acceptance check; do not start the next phase until the check passes.

| Phase | What you build | Rough time |
|---|---|---|
| 0 | Tools, folders, Python and Java environments | half a day |
| 1 | Shared foundations: config, seeds, data schema, logging | 1 day |
| 2 | Full pipeline in a fast synthetic Python environment (the prototype) | 1.5 to 2 weeks |
| 3 | CloudSim Plus simulator, Python bridge, retraining on simulator data | 2 to 3 weeks |
| 4 | Experiment grid: baselines, ablations, workloads, seeds | 1 week (mostly compute) |
| 5 | Analysis, statistics, figures, final report | 1 week |

The prototype is not throwaway. Every Python module written in Phase 2 (models, uncertainty, decision layer, metrics) is reused unchanged in Phase 3; only the environment changes from a formula to CloudSim Plus.

Conventions used below:

* Commands are given for **Windows PowerShell** first, because the project lives under `C:\projects`. Linux and macOS equivalents follow where they differ.
* `PS>` marks a PowerShell prompt; do not type it.
* Paths are relative to the project root `C:\projects\uq-autoscaler` unless they start with a drive letter.
* "Acceptance check" means a test or script that must pass before you move on.

# Phase 0: Tools and environment

## Step 0.1: Install the base tools

You need Git, Python 3.11, a Java Development Kit (JDK) 21 and Apache Maven. JDK 21 is a long-term-support release and satisfies CloudSim Plus 8.x. Do not use the CloudSim Plus `master` branch or `9.0.0-SNAPSHOT`: its site says those need a newer JDK, and snapshots change without notice, which breaks reproducibility.

Open PowerShell **as a normal user** and run:

```powershell
PS> winget install -e --id Git.Git
PS> winget install -e --id Python.Python.3.11
PS> winget install -e --id EclipseAdoptium.Temurin.21.JDK
```

Close and reopen PowerShell so the new `PATH` entries load.

Maven is installed by hand:

1. Download the latest **3.9.x binary zip** from `https://maven.apache.org/download.cgi` (the file is named `apache-maven-3.9.<n>-bin.zip`).
2. Extract it to `C:\tools\` so that `C:\tools\apache-maven-3.9.<n>\bin\mvn.cmd` exists.
3. Open *Start*, search for "Edit the system environment variables", then click *Environment Variables*. Under *User variables*:
    * add `MAVEN_HOME` = `C:\tools\apache-maven-3.9.<n>`;
    * edit `Path` and add a new entry `%MAVEN_HOME%\bin`;
    * check that `JAVA_HOME` exists and points to the Temurin 21 folder (for example `C:\Program Files\Eclipse Adoptium\jdk-21.<...>-hotspot`). If it is missing, add it.
4. Reopen PowerShell.

Do not use `setx PATH ...` to edit the path: `setx` truncates values longer than 1024 characters and can silently destroy your `PATH`.

Linux (Debian/Ubuntu) equivalent:

```bash
sudo apt update
sudo apt install -y git python3.11 python3.11-venv openjdk-21-jdk maven
```

macOS (Homebrew):

```bash
brew install git python@3.11 openjdk@21 maven
```

Optional but useful: Visual Studio Code with the *Python* and *Extension Pack for Java* extensions, or IntelliJ IDEA Community for the Java part.

**Acceptance check 0.1.** Each command prints a version:

```powershell
PS> git --version
PS> py -3.11 --version        # Python 3.11.x
PS> java -version             # openjdk version "21..."
PS> javac -version            # javac 21...
PS> mvn -v                    # Apache Maven 3.9.x ... Java version: 21...
```

If `mvn -v` reports a Java version other than 21, `JAVA_HOME` points at the wrong JDK. Fix it before continuing.

## Step 0.2: Create the project folders

```powershell
PS> mkdir C:\projects\uq-autoscaler
PS> cd C:\projects\uq-autoscaler
PS> $dirs = @(
  "configs", "docs",
  "src\autoscaler\env", "src\autoscaler\data", "src\autoscaler\models",
  "src\autoscaler\uncertainty", "src\autoscaler\decision", "src\autoscaler\sim",
  "src\autoscaler\bridge", "src\autoscaler\eval",
  "scripts", "tests",
  "cloudsim\src\main\java\org\uqautoscaler\sim",
  "cloudsim\src\test\java\org\uqautoscaler\sim",
  "data\raw", "data\processed", "models",
  "results\runs", "results\figures", "results\tables", "results\reports"
)
PS> $dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }
PS> "env","data","models","uncertainty","decision","sim","bridge","eval" |
      ForEach-Object { New-Item -ItemType File -Force "src\autoscaler\$_\__init__.py" | Out-Null }
PS> New-Item -ItemType File -Force src\autoscaler\__init__.py, tests\__init__.py | Out-Null
```

Linux/macOS: the same tree with `mkdir -p` and `touch`.

The finished layout looks like this (files are created in later steps):

```text
uq-autoscaler/
  README.md                project overview and how to reproduce
  .gitignore
  pyproject.toml           makes src/autoscaler an installable package
  requirements.txt         Python dependencies (minimum versions)
  requirements.lock.txt    exact versions, written by pip freeze
  configs/
    prototype.yaml         synthetic environment and model settings
    cloudsim.yaml          CloudSim Plus scenario settings
    experiments.yaml       experiment grid: workloads, seeds, policies
  docs/
    ROADMAP.md             this document
  src/autoscaler/
    config.py  seeds.py  logging_utils.py  schema.py
    env/        workloads.py  synthetic.py
    data/       calibration.py  dataset.py  splits.py
    models/     features.py  metric_filters.py  metric_estimator.py
                rt_ensemble.py  quantile.py  tuning.py
    uncertainty/ propagation.py  calibration_metrics.py  conformal.py
    decision/   power.py  objectives.py  pareto.py  selection.py  policies.py
    sim/        loop.py
    bridge/     protocol.py  server.py
    eval/       metrics.py  stats.py  plots.py  report.py
  scripts/      numbered entry points (p2_..., p3_..., p4_..., p5_...)
  tests/        pytest unit and integration tests
  cloudsim/     Maven project (Java 21, CloudSim Plus)
  data/         raw and processed datasets (not committed)
  models/       trained models (not committed)
  results/      runs, figures, tables, reports
```

## Step 0.3: Initialise Git

```powershell
PS> git init
PS> git branch -M main
```

Create `.gitignore` with this content:

```text
# Python
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
# Java
cloudsim/target/
*.class
.idea/
.vscode/
# Data and outputs (regenerated by scripts)
data/raw/
data/processed/
models/
results/runs/
```

Figures, tables and reports stay committed so a reader can see results without re-running.

```powershell
PS> git add .gitignore
PS> git commit -m "Project skeleton"
```

## Step 0.4: Python virtual environment and packages

```powershell
PS> py -3.11 -m venv .venv
PS> .\.venv\Scripts\Activate.ps1
```

If PowerShell refuses with "running scripts is disabled on this system", run once:

```powershell
PS> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

and activate again. Your prompt now starts with `(.venv)`. Activate the environment in every new terminal before working on the project. (Linux/macOS: `python3.11 -m venv .venv` then `source .venv/bin/activate`.)

Upgrade pip and install PyTorch (CPU build; the networks are small and a GPU is not needed):

```powershell
(.venv) PS> python -m pip install --upgrade pip
(.venv) PS> pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Create `requirements.txt`:

```text
numpy>=1.26
pandas>=2.1
pyarrow>=14
scipy>=1.11
scikit-learn>=1.4
matplotlib>=3.8
pyyaml>=6.0
joblib>=1.3
tqdm>=4.66
pytest>=8.0
```

`scikit-learn` provides linear models, gradient-boosted trees (including the quantile loss used in an ablation) and cross-validation; `scipy` provides the t-test and the Wilcoxon signed-rank test; `pyarrow` lets pandas read and write Parquet, which is much faster than CSV for large run logs.

```powershell
(.venv) PS> pip install -r requirements.txt
```

Create `pyproject.toml` so `import autoscaler` works from scripts and tests:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "autoscaler"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```powershell
(.venv) PS> pip install -e .
(.venv) PS> pip freeze > requirements.lock.txt
```

`pip install -e .` installs the package in *editable* mode: edits in `src/autoscaler` take effect without reinstalling.

**Acceptance check 0.4.**

```powershell
(.venv) PS> python -c "import torch, sklearn, scipy, pandas, yaml, autoscaler; print(torch.__version__, sklearn.__version__)"
(.venv) PS> pytest -q        # "no tests ran" is fine at this point
```

## Step 0.5: Create the Maven project for CloudSim Plus

Create `cloudsim\pom.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
                             http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.uqautoscaler</groupId>
  <artifactId>uq-autoscaler-sim</artifactId>
  <version>0.1.0</version>
  <packaging>jar</packaging>

  <properties>
    <maven.compiler.release>21</maven.compiler.release>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  </properties>

  <dependencies>
    <dependency>
      <groupId>org.cloudsimplus</groupId>
      <artifactId>cloudsimplus</artifactId>
      <version>8.5.7</version>
    </dependency>
    <dependency>
      <groupId>com.fasterxml.jackson.core</groupId>
      <artifactId>jackson-databind</artifactId>
      <version>2.17.2</version>
    </dependency>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>5.10.2</version>
      <scope>test</scope>
    </dependency>
  </dependencies>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.2.5</version>
      </plugin>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-shade-plugin</artifactId>
        <version>3.5.3</version>
        <executions>
          <execution>
            <phase>package</phase>
            <goals><goal>shade</goal></goals>
            <configuration>
              <transformers>
                <transformer implementation=
                  "org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                  <mainClass>org.uqautoscaler.sim.Main</mainClass>
                </transformer>
              </transformers>
            </configuration>
          </execution>
        </executions>
      </plugin>
    </plugins>
  </build>
</project>
```

`8.5.7` was the latest final CloudSim Plus release on Maven Central when this roadmap was written; pin it and do not use `LATEST`, so every run uses the same simulator. Jackson reads and writes JSON for the Python bridge. The shade plugin builds one runnable jar with all dependencies inside.

Create a smoke-test class `cloudsim\src\main\java\org\uqautoscaler\sim\Main.java`:

```java
package org.uqautoscaler.sim;

import org.cloudsimplus.core.CloudSimPlus;

public class Main {
    public static void main(String[] args) {
        CloudSimPlus sim = new CloudSimPlus();
        System.out.println("CloudSim Plus engine created; clock = " + sim.clock());
    }
}
```

```powershell
PS> cd cloudsim
PS> mvn -q package
PS> java -jar target\uq-autoscaler-sim-0.1.0.jar
PS> cd ..
```

**Acceptance check 0.5.** The jar prints `CloudSim Plus engine created; clock = 0.0`. If Maven reports `class file has wrong version` or `release version 21 not supported`, your `JAVA_HOME` is not JDK 21. If it says a class needs a newer Java than 21, install the version it names and set `JAVA_HOME` to it.

Commit:

```powershell
PS> git add . ; git commit -m "Environments and Maven skeleton"
```

# Phase 1: Shared foundations

These pieces are used by every later phase. Build them first so the prototype and the CloudSim Plus version share one configuration system, one data format and one seeding scheme.

## Step 1.1: Configuration files

All numbers live in YAML files under `configs/`. Code never hard-codes a constant that a reader might want to change.

Create `configs/prototype.yaml`:

```yaml
time:
  interval_s: 60            # one control interval = one minute
  steps_per_run: 1440       # 24 hours
  warmup_steps: 200         # excluded from metrics
system:
  vm_min: 1
  vm_max: 30
  initial_vms: 4
  service_rate_per_vm: 25.0   # requests/s one VM can serve (mu)
  base_latency_ms: 40.0       # L0, service time with no queueing
  boot_delay_steps: 1         # a new VM serves only after this many steps
  complexity: {amplitude: 0.12, period_steps: 260, noise_std: 0.03}
  rt_noise_frac: 0.05
slo:
  rt_ms: 250.0
  delta: 0.05                 # allowed violation probability per decision
actions: {k_min: -7, k_max: 7}
cost: {price_per_vm_hour: 0.10}
power: {model: square, p_idle_w: 80.0, p_max_w: 400.0}   # linear|square|cubic|sqrt
calibration: {rows: 20000, rate_min: 30.0, rate_max: 460.0, seed: 7}
splits: {train: 0.6, calib: 0.2, test: 0.2, seed: 11}
metric_filter: {r2_threshold: 0.90, ttest_alpha: 0.05}
stage1: {candidates: [paper_linear, basis_linear, hgb, mlp_ensemble], cv_folds: 5}
ensemble:
  members: 5
  val_frac: 0.15
  max_epochs: 300
  patience: 20
  search_trials: 20
  include_request_rate: false
propagation: {draws: 32}
selection: {rule: aug_tchebycheff, weights: [1.0, 1.0, 1.0], rho: 0.01}
```

Create `configs/experiments.yaml` (Phase 4 extends it):

```yaml
eval_seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
calibration_seeds: [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
workloads: [daily_ramp_burst, step, flash_crowd, sawtooth]
baselines:
  cpu_threshold: {upper: 80.0, lower: 30.0, cooldown_steps: 3}
  rt_threshold:  {upper_frac_slo: 0.8, lower_frac_slo: 0.4, cooldown_steps: 3}
```

The threshold values are placeholders. Replace them with the values the base paper used for its CPU and response-time baselines, and note the source in a comment.

Write `src/autoscaler/config.py`: a function `load_config(path) -> dict` that reads YAML, plus `get(cfg, "a.b.c")` for dotted access. Keep it simple; plain dicts are fine. Every script takes `--config` and `--out` arguments and copies the config it used into its output folder, so every result can be traced to its settings.

## Step 1.2: Seeds and reproducibility

`src/autoscaler/seeds.py`:

```python
import random, numpy as np, torch

def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)
```

Rules that keep runs reproducible:

* Pass a `numpy.random.Generator` into every function that draws random numbers. Never call the global `np.random` inside library code.
* Derive child seeds from a parent seed with `np.random.SeedSequence(parent).spawn(n)`, so the workload, the system noise and the controller's Monte Carlo draws each get their own independent stream. Changing the controller then does not change the workload.
* Calibration seeds (1001 to 1010) and evaluation seeds (1 to 10) never overlap.

## Step 1.3: Data schema

Two tables carry everything. Define their columns once in `src/autoscaler/schema.py` as Python lists so writers and readers agree.

**Transition table** (training data for both model stages). One row is one scaling transition:

| Column | Meaning |
|---|---|
| `source`, `run_id`, `step` | where the row came from (synthetic or cloudsim; run; interval index) |
| `n_before`, `k`, `n_after` | pool before, action, pool after (after clamping) |
| `rate_before`, `rate_after` | arrivals per second in each interval (recorded, not a default model input) |
| `<metric>_before` for each metric | telemetry in the interval before the action |
| `<metric>_after` for each metric | telemetry in the interval after the action |
| `rt_after_ms` | mean response time in the interval after the action (the stage-2 target) |

**Run log** (one row per control interval of a closed-loop run): `run_id, policy, workload, seed, step, rate, n_running, n_booting, k, fallback, rt_mean_ms, rt_p95_ms, slo_violated, cost_usd, energy_wh`, plus every metric. A separate **decision log** (JSON lines) stores, for each decision, every candidate's `n`, `mean_log`, `sigma`, `U`, `cost`, `energy`, `feasible`, `on_front`, `score`, and the chosen action.

Save tables as Parquet (`df.to_parquet(path)`); they load ten times faster than CSV.

**Definition used throughout:** an interval *violates the SLO* if the **mean** response time of requests completed in that interval exceeds `slo.rt_ms`. Use the same definition in Python and Java.

## Step 1.4: Logging helper

`src/autoscaler/logging_utils.py`: a function `run_dir(base, name)` that creates `results/runs/<timestamp>_<name>/`, writes `config.yaml` and the current `git rev-parse HEAD` into it, and returns the path. Every script writes only inside its run directory.

**Acceptance check 1.** `tests/test_foundations.py` checks that `load_config` reads both YAML files, that two calls of `seed_everything(5)` followed by `rng.normal(size=3)` give identical numbers, and that the schema column lists contain no duplicates. Run `pytest -q`.

# Phase 2: The prototype in a synthetic environment

The prototype runs the whole planned pipeline against a formula-based system that simulates a day in under a second. That makes it possible to debug the models and the decision layer, and to run every baseline and ablation, before CloudSim Plus enters the picture.

## Step 2.1: Workloads

`src/autoscaler/env/workloads.py` defines `make_workload(shape: str, steps: int, rng) -> np.ndarray` returning request rates in requests per second, clipped to `[30, 460]`.

Shapes are defined on the fraction $x=t/N$ of the run, so the same shape works for a 1440-step synthetic day and a shorter CloudSim Plus run.

| Shape | Formula | What it tests |
|---|---|---|
| `daily_ramp_burst` | $120 + 60\sin(2\pi x) + 150\,\mathrm{clip}(\frac{x-0.208}{0.347},0,1) + 180\,e^{-\frac12(\frac{x-0.729}{0.0278})^2} + \varepsilon_t$ | the shape from the notes (ramp over steps 300 to 800 and burst at step 1050 when $N=1440$) |
| `step` | 120 for $x<0.417$, then 330, plus $\varepsilon_t$ | reaction to a sudden lasting jump |
| `flash_crowd` | 120 plus three Gaussian spikes of height 250 and width $0.0104N$ steps at random times in $x\in[0.1,0.9]$, plus $\varepsilon_t$ | the fallback and the action range |
| `sawtooth` | rises linearly from 100 to 380 over $N/6$ steps, drops back, repeats; plus $\varepsilon_t$ | churn |
| `random_walk` | $\lambda_0=150$, $\lambda_t=\lambda_{t-1}+\mathcal N(0,6^2)$, reflected at 30 and 460 | used only for calibration in Phase 3 |

with $\varepsilon_t\sim\mathcal N(0,8^2)$ in every shape. Spike times and noise come from `rng`, so the same seed always gives the same workload.

**Acceptance check 2.1.** A test generates each shape twice with the same seed (identical arrays) and with different seeds (different arrays), and checks all values lie in `[30, 460]`. A script `scripts/p2_00_plot_workloads.py` saves `results/figures/workloads.png` with all shapes; look at it.

## Step 2.2: The synthetic system

`src/autoscaler/env/synthetic.py` defines a class `SyntheticSystem(cfg, rng)` with:

* `complexity(t)`: $c_t=\max\big(0.5,\,1+0.12\sin\frac{2\pi t}{260}+\eta_t\big)$, $\eta_t\sim\mathcal N(0,0.03^2)$. Hidden from all models.
* `observe(rate, n_serving, t) -> dict`: utilisation $u=\mathrm{clip}\big(\frac{\lambda c_t}{n\mu},0,0.995\big)$; response time $\max\big(20,\frac{L_0}{1-u}(1+\xi)\big)$ with $\xi\sim\mathcal N(0,0.05^2)$; and the telemetry

| Metric | Definition |
|---|---|
| `cpu_util` | $\mathrm{clip}(100u+\mathcal N(0,1.5^2),0,100)$ |
| `mem_util` | $\mathrm{clip}(30+45u+\mathcal N(0,2^2),0,100)$ |
| `net_in_kbps` | $\max(0,8\lambda+\mathcal N(0,20^2))$ |
| `disk_iops` | $\max(0,12\lambda/n+\mathcal N(0,3^2))$ |
| `queue_len` | $\max\big(0,\frac{u}{1-u}+\mathcal N(0,0.15^2)\big)$ |

* A pool state: `n_running` and a list of booting VMs with the step at which each becomes ready (`boot_delay_steps`). `apply_action(k, t)` adds VMs to the booting list or removes running VMs immediately (scale-in removes booting VMs first). Booting VMs are billed and draw idle power but serve nothing.

**Acceptance check 2.2.** Unit tests: response time rises monotonically with rate at fixed pool (average over 200 noisy samples); at $u=0.84$ the noiseless response time equals 250 ms; a VM added at step $t$ serves from step $t+1$; scale-in never drops below `vm_min`.

## Step 2.3: Calibration data (transitions)

`src/autoscaler/data/calibration.py`, function `collect_synthetic(cfg, rng) -> DataFrame`. For each of `calibration.rows` rows:

1. Draw $\lambda\sim\mathrm{Uniform}(30,460)$, a time $t\sim\mathrm{Uniform}\{0,\dots,1439\}$ (for the complexity term), $n_{before}\sim\mathrm{Uniform}\{1,\dots,30\}$ and $k\sim\mathrm{Uniform}\{-7,\dots,7\}$.
2. Observe at $n_{before}$ (before-metrics), then at $n_{after}=\mathrm{clip}(n_{before}+k,1,30)$ with the same $\lambda$ and $t$ (after-metrics and `rt_after_ms`). Scaling does not change how many requests arrive, so the rate is the same on both sides.

Sampling $\lambda$ uniformly over its whole range, instead of from an evaluation workload, is what keeps training data separate from test data.

`scripts/p2_01_generate_calibration.py --config configs/prototype.yaml` writes `data/processed/synthetic_transitions.parquet`.

`src/autoscaler/data/splits.py`: `random_split(df, fracs, seed)` returning train, calibration and test frames (60/20/20). Synthetic rows are independent, so a random split is valid. For CloudSim Plus data (Phase 3) use `group_split(df, "run_id", ...)`, which keeps every row of a run on one side, because rows within a run are correlated in time.

**Acceptance check 2.3.** The parquet file has 20000 rows, no missing values, and `k` covers $-7$ to $7$. Print the share of rows with `rt_after_ms > 250`; it should be well above zero (roughly a quarter or more), otherwise the models never see the region that matters.

## Step 2.4: Metric selection filters (base paper)

`src/autoscaler/models/metric_filters.py`:

* `ttest_filter(df, metrics, alpha)`: for each metric compute $\Delta=m_{after}-m_{before}$; split rows into $k>0$ and $k<0$; run Welch's t-test (`scipy.stats.ttest_ind(a, b, equal_var=False)`). Keep the metric if $p<\alpha$. Rationale: a metric that moves the same way after scale-up and scale-down carries no information about the action.
* `r2_filter(scores, threshold)`: keep a metric if its best stage-1 model's cross-validated $R^2$ exceeds 0.90. Rationale: at decision time stage 2 is fed stage-1 *estimates*; a metric that cannot be estimated well only adds noise.

The final metric set is the intersection of both filters. Save it to `models/selected_metrics.json`.

## Step 2.5: Stage 1, the non-linear metric estimator (E1)

`src/autoscaler/models/features.py` builds stage-1 inputs for metric $m$ with current pool $w$ and action $k$ (with $n=\max(w+k,1)$):

* paper features: $\big[m\frac{w}{n},\ m\frac{k}{n}\big]$ (paper Eq. 1);
* basis features: $\big[m,\ w,\ k,\ m\frac{w}{n}\big]$.

`src/autoscaler/models/metric_estimator.py` implements one estimator per metric with four candidate model types:

| Name | Model | Role |
|---|---|---|
| `paper_linear` | `LinearRegression` on paper features | the base paper's estimator; used by the OPT baseline |
| `basis_linear` | `LinearRegression` on basis features | strong linear reference |
| `hgb` | `HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)` on basis features | non-linear candidate |
| `mlp_ensemble` | 5 small MLPs (32, 16), average of members | non-linear candidate |

Procedure, per metric:

1. 5-fold cross-validation on the training split for every candidate; record the mean $R^2$.
2. Choose the candidate with the highest $R^2$ (ties go to the simpler model). Refit it on the whole training split.
3. Keep the **out-of-fold** predictions of the chosen model and compute residuals $e=m_{after}-\hat m_{after}$ for every training row.
4. After all metrics are done, form the residual matrix (rows by selected metrics) and its covariance $\hat\Sigma_e$ with `np.cov(E, rowvar=False)`. Also keep the raw residual matrix for resampling.

Out-of-fold residuals matter: residuals on rows the model was fitted on are too small and would make the propagated uncertainty too optimistic.

API: `MetricEstimator.fit(df, metrics)`, `.predict(before: dict, w: int, ks: array) -> array (len(ks), n_metrics)`, `.residual_cov`, `.residuals`, `.save(path)`, `.load(path)`. Clip predicted percentages to `[0, 100]` and counts to `>= 0`.

Script `scripts/p2_02_fit_stage1.py` runs the t-test filter, fits all candidates, applies the $R^2$ filter, and writes `models/stage1.joblib` plus `results/tables/stage1_r2.csv` (metric by candidate).

**Acceptance check 2.5.** `queue_len` should either be rejected by the linear candidates and accepted only if a non-linear candidate clears 0.90, or be rejected outright; `disk_iops` should score near 1.0 with `basis_linear`. The table shows non-linear candidates at least matching the linear ones on every metric.

## Step 2.6: Stage 2, the deep ensemble for response time (E1, E2)

`src/autoscaler/models/rt_ensemble.py`, in PyTorch.

**One member.** Inputs: the selected post-scaling metrics and $n_{after}$ (plus `rate_after` only if `ensemble.include_request_rate` is true; the default is false, matching the base paper's final model). Inputs are standardised with mean and standard deviation from the training split. Target $y=\log(\mathrm{rt\_after\_ms})$.

```python
class Member(nn.Module):
    def __init__(self, d_in, h1, h2, dropout, slope):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, h1), nn.LeakyReLU(slope), nn.Dropout(dropout),
            nn.Linear(h1, h2),   nn.LeakyReLU(slope), nn.Dropout(dropout))
        self.mean = nn.Linear(h2, 1)
        self.var_raw = nn.Linear(h2, 1)
    def forward(self, x):
        h = self.body(x)
        mu = self.mean(h).squeeze(-1)
        var = nn.functional.softplus(self.var_raw(h)).squeeze(-1) + 1e-6
        return mu, var
```

Two hidden layers with LeakyReLU and dropout is the base paper's architecture, so the ablation "one member, mean only" reproduces the paper's predictor.

**Loss.** Gaussian negative log-likelihood, `torch.nn.GaussianNLLLoss()(mu, y, var)`, which computes $\frac12\log\sigma^2+\frac{(y-\mu)^2}{2\sigma^2}$ averaged over the batch. Optimiser Adam, mini-batches, up to `max_epochs`, early stopping on validation NLL with `patience`, restoring the best weights.

**Ensemble training.** For member $j=1..M$ with seed `base + j`:

1. Split off a validation slice (`val_frac`) from the training split **first**.
2. Draw a bootstrap sample (with replacement) of the remaining rows.
3. Train on the bootstrap sample; early-stop on the validation slice.

Drawing the validation slice before bootstrapping prevents validation rows from also appearing in training.

**Hyperparameter search** (`src/autoscaler/models/tuning.py`): 20 random trials for a single member, scored by validation NLL, over:

| Hyperparameter | Range |
|---|---|
| hidden sizes $(h_1,h_2)$ | (32,16), (64,32), (128,64) |
| learning rate | log-uniform $[10^{-4}, 3\cdot10^{-3}]$ |
| dropout | 0.0, 0.1, 0.2 |
| weight decay | log-uniform $[10^{-6}, 10^{-3}]$ |
| batch size | 64, 128, 256 |
| LeakyReLU slope | 0.01, 0.1 |

Save the best setting to `models/rt_hparams.json` and use it for all members.

**Prediction.** `predict(X) -> (mu, var)` with arrays of shape `(M, rows)`, members in `eval()` mode (dropout off).

Script `scripts/p2_03_train_rt.py` runs the search, trains the ensemble, and trains one more network of the same architecture with a single output and mean squared error loss (the base paper's predictor, used by the OPT baseline). It saves both to `models/`. With `--include-rate` it trains an extra ensemble that also takes `rate_after` as an input and saves it to `models/with_rate/`, for the `abl_with_rate` ablation.

**Acceptance check 2.6.** On the test split with *true* post-scaling metrics as inputs, ensemble-mean $R^2$ on $\log$ response time is above 0.9, and MAE in milliseconds on rows with response time below 500 ms is reported. Training is deterministic: re-running with the same seeds gives the same test score to four decimals.

## Step 2.7: Quantile model for the ablation (E2)

`src/autoscaler/models/quantile.py`:

* `QuantileNet`: the same body as `Member`, one output, trained with the pinball loss at $\tau=1-\delta=0.95$ on $\log$ response time:

```python
def pinball(y, q, tau):
    r = y - q
    return torch.mean(torch.maximum(tau * r, (tau - 1) * r))
```

* `QuantileHGB`: `HistGradientBoostingRegressor(loss="quantile", quantile=0.95)` on the same inputs, as a non-neural reference.

Feasibility under this ablation is $e^{\hat q_{0.95}}\le\mathrm{SLO}$.

**Acceptance check 2.7.** On the test split, the share of rows with $y\le\hat q_{0.95}$ is between 0.93 and 0.97 for both models.

## Step 2.8: Uncertainty propagation and the bound (E2)

`src/autoscaler/uncertainty/propagation.py`, function:

```python
def candidate_bounds(stage1, ensemble, before, w, ks, z, S, rng, mode="full"):
    """Return per-candidate dict arrays: n, mean_log, sigma, sigma_ale,
    sigma_epi, sigma_stage1, U (ms)."""
```

For each candidate $k$:

1. $\hat{\mathbf m}'(k)=$ `stage1.predict(before, w, [k])`.
2. Draw $S$ error vectors $\mathbf e^{(s)}$. Default: Gaussian, $\mathbf e=L\mathbf z$ with $L$ the Cholesky factor of $\hat\Sigma_e$ (`np.linalg.cholesky`, adding $10^{-9}I$ if it fails) and $\mathbf z\sim\mathcal N(0,I)$. Alternative (`resample=True`): pick $S$ rows of the stored residual matrix at random.
3. Form $S$ inputs $\hat{\mathbf m}'(k)+\mathbf e^{(s)}$, clip to valid ranges, and run **each** member on **each** input. Never average the inputs first.
4. Pool the $M\times S$ outputs $(m_j^{(s)},s_j^{2(s)})$:

$$\bar m_k=\frac{1}{MS}\sum_{j,s}m_j^{(s)},\qquad \sigma_k^2=\underbrace{\frac{1}{MS}\sum_{j,s}s_j^{2(s)}}_{\text{aleatoric}}+\underbrace{\frac{1}{MS-1}\sum_{j,s}\big(m_j^{(s)}-\bar m_k\big)^2}_{\text{epistemic + stage 1}}$$

5. Bound: $U_k=\exp(\bar m_k+z\,\sigma_k)$ with $z=\Phi^{-1}(1-\delta)=1.6449$ (or the conformal value from Step 2.9).

Also report the parts separately: `sigma_ale` from the first sum; `sigma_epi` as the mean over draws of the across-member variance; `sigma_stage1` as the variance across draws of the per-draw member average. Compute these two with population divisors (divide by $M$ and by $S$); then, by the law of total variance, `sigma_epi`$^2$ + `sigma_stage1`$^2$ equals exactly the population variance of all $M\cdot S$ means.

The `mode` argument implements the ablations: `"none"` ($\sigma=0$), `"epistemic"` (only member disagreement at the central estimate, $S=1$, no aleatoric term), `"ale_epi"` (both terms, $S=1$), `"full"` (everything above).

Use one batched forward pass per member over all $|\mathcal K|\cdot S$ inputs; with 15 candidates, 32 draws and 5 members this is 5 passes over 480 rows.

**Acceptance check 2.8.** Tests: with $\hat\Sigma_e=0$ the `full` result equals `ale_epi`; with one member and zero aleatoric variance $\sigma=0$; increasing $\hat\Sigma_e$ never decreases $\sigma_k$; `sigma_epi`$^2$ + `sigma_stage1`$^2$ equals the population variance of the $M\cdot S$ means within floating-point tolerance.

## Step 2.9: Calibration evaluation and conformal correction (E2)

`src/autoscaler/uncertainty/calibration_metrics.py` computes, on held-out transitions passed **end to end** (stage-1 estimates as inputs, never the true after-metrics):

| Measure | Formula |
|---|---|
| coverage | $\frac1N\sum_i\mathbf 1\{y_i\le U_i\}$ |
| mean margin | $\frac1N\sum_i(U_i-e^{\bar m_i})$ in ms |
| conditional coverage | coverage in 5 bins of $e^{\bar m_i}$ (quantile bins), especially the bin containing the SLO |
| NLL | $\frac1N\sum_i\big[\frac12\log(2\pi\sigma_i^2)+\frac{(\log y_i-\bar m_i)^2}{2\sigma_i^2}\big]$ |
| pinball loss | for the quantile models |

`src/autoscaler/uncertainty/conformal.py`: on the calibration split compute scores $r_i=(\log y_i-\bar m_i)/\sigma_i$, sort them and take $z_{cal}=r_{(\lceil (n_c+1)(1-\delta)\rceil)}$. If `use_conformal: true` the controller uses $z_{cal}$ instead of 1.6449.

Script `scripts/p2_04_calibration_report.py` evaluates every uncertainty mode and both quantile models on the test split and writes `results/tables/calibration.csv` and a reliability figure (nominal vs observed coverage at $\tau\in\{0.5,0.8,0.9,0.95,0.99\}$).

**Acceptance check 2.9.** The `full` mode reaches coverage within 0.93 to 0.97 at the 0.95 level; `epistemic` alone is clearly below it (this reproduces the earlier prototype finding). If `full` is outside the band, enable conformal correction and confirm it brings coverage into the band on the test split.

## Step 2.10: Objectives, power models and the Pareto layer (E3)

`src/autoscaler/decision/power.py`:

$$P(u)=P_{idle}+(P_{max}-P_{idle})\,h(u),\qquad h(u)\in\{u,\ u^2,\ u^3,\ \sqrt u\}$$

`src/autoscaler/decision/objectives.py`, for each candidate:

* latency $f_L=U_k$ (ms);
* cost $f_C=n_k\cdot p\cdot\Delta t_h$, where $n_k$ counts running plus booting VMs after the action and $\Delta t_h$ is the interval in hours;
* energy $f_E=n_k\cdot P(\hat u_k)\cdot\Delta t_h$ (Wh), with $\hat u_k=$ predicted post-scaling `cpu_util`/100 from stage 1 (one machine per VM in the synthetic setting).

`src/autoscaler/decision/pareto.py`: `pareto_mask(F)` for an array of shape `(candidates, objectives)`, all minimised: a row is dominated if another row is $\le$ in every column and $<$ in at least one.

`src/autoscaler/decision/selection.py`:

1. Feasible set $\mathcal F=\{k:U_k\le\mathrm{SLO}\}$. If empty, return the largest pool (base paper fallback) and flag `fallback=True`.
2. Front $\mathcal P$ = non-dominated rows of $\mathcal F$ over $(f_L,f_C,f_E)$.
3. Normalise each objective over $\mathcal P$: $\tilde f_i=\frac{f_i-\min f_i}{\max f_i-\min f_i}$ (0 if all equal).
4. Score with augmented Tchebycheff: $\max_i w_i\tilde f_i+\rho\sum_i w_i\tilde f_i$, $\rho=0.01$. Pick the lowest score; break exact ties by smaller $|k|$, then smaller pool.

Other rules (`min_cost`, `weighted_sum`, `min_energy`) are implemented in the same module for ablations.

**Acceptance check 2.10.** `tests/test_decision.py` must include:

* the worked example from the notes at $\lambda=150$ ($n=8..12$, square model, $f_L=40/(1-u)$): all five on the front; augmented Tchebycheff picks $n=10$; weighted sum picks $n=11$; `min_cost` picks $n=8$;
* a proof-by-test that `min_cost` always returns the smallest feasible pool on 1000 random candidate sets;
* the linear power model makes energy increase with $n$;
* the fallback fires when every candidate is infeasible;
* a dominated candidate is never chosen by augmented Tchebycheff.

## Step 2.11: Policies

`src/autoscaler/decision/policies.py`. Every policy implements `decide(obs: dict, state: dict) -> (k, info)` where `obs` holds the last interval's metrics, `n_running`, `n_booting` and measured `rt_mean_ms`.

| Policy name | Rule |
|---|---|
| `cpu_threshold` | if mean CPU > upper: $k=+1$; if < lower: $k=-1$; else 0; no action during cooldown |
| `rt_threshold` | same with measured response time against fractions of the SLO |
| `opt` | `paper_linear` stage 1 + the single MSE-trained network from Step 2.6; scan $k$ from $k_{min}$ upward; first $k$ with $e^{\hat y}<\mathrm{SLO}$; otherwise $k_{max}$ |
| `proposed` | full pipeline: chosen stage 1, ensemble, `full` propagation, chance constraint, augmented Tchebycheff |
| `abl_*` | `proposed` with one component switched, named after it (Step 4.2 lists them) |

Decisions use the pool that will exist after pending boots: $w=n_{running}+n_{booting}$.

## Step 2.12: The closed loop

`src/autoscaler/sim/loop.py`, `run_episode(policy, system, workload, cfg, seed) -> (run_log, decision_log)`: for every step, observe the system, compute cost and energy for the step from the **true** utilisation (realised values, not predictions), let the policy decide every `decision_every` steps (default 1), apply the action, and append to both logs.

Script `scripts/p2_05_run_prototype.py --policy proposed --workload daily_ramp_burst --seed 1` runs one episode and saves the logs under `results/runs/`.

**Acceptance check 2.12.** One `proposed` episode on `daily_ramp_burst` runs in under a minute. Plot the pool size against the request rate and the response time against the SLO; the pool should follow the load with a lag of at most a few steps.

## Step 2.13: Prototype experiment sweep and sanity checks

Script `scripts/p2_06_sweep.py` runs every policy and ablation (Step 4.2) on every workload and evaluation seed in the synthetic environment, with a process pool for speed (`concurrent.futures.ProcessPoolExecutor`), and then calls the Phase 5 analysis on the results with `--env synthetic`.

**Acceptance check 2 (end of prototype).** All of the following hold, or the reason they do not is understood and written down in `docs/prototype_findings.md`:

1. `proposed` realised violation rate is lower than `opt`'s on at least three of four workloads.
2. `abl_min_cost` (the `proposed` pipeline with the `min_cost` rule) chooses the smallest feasible pool in every non-fallback decision (this reproduces the inertness result).
3. Coverage of `full` is in the 0.93 to 0.97 band end to end.
4. The power-model sensitivity reproduces the analytic threshold: recompute the two-objective (cost, energy) front of the feasible candidates from the decision logs; with the square model and $\Delta P/P_{idle}=4$ it has more than one point in many decisions, and with $\Delta P/P_{idle}=1$ (below the 1.42 threshold) it almost always has exactly one.
5. All tests pass and two runs with the same seed produce identical logs (compare file hashes).

Commit and tag: `git tag prototype-complete`.

# Phase 3: The CloudSim Plus implementation

Phase 3 replaces the formula-based system with a discrete-event simulation. Python keeps everything that learns or decides; Java simulates the data centre and asks Python for a decision every interval through a local socket.

```text
          Python (src/autoscaler)                     Java (cloudsim/)
 +---------------------------------------+     +-------------------------------+
 | bridge/server.py                      |<--->| RemotePolicy (TCP client)     |
 |  - loads stage 1, ensemble, policies  | JSON| ControlLoop (clock listener)  |
 |  - runs propagation + Pareto layer    |lines| Cluster: hosts, VMs, broker   |
 |  - writes decision log                |     | ArrivalGenerator (Poisson)    |
 +---------------------------------------+     | Telemetry, Energy, RunLogger  |
                                               +-------------------------------+
```

Why a socket bridge and not ONNX inside Java: the Monte Carlo propagation, the uncertainty modes, the quantile ablation and the decision rules already exist in Python from Phase 2 and are unit-tested there. Re-implementing them in Java would duplicate code and create two places for bugs. Python inference in `eval()` mode is deterministic, so runs stay reproducible. If runs turn out too slow, exporting the members to ONNX is an optional later optimisation, not part of the plan.

## Step 3.1: Scenario and its numbers

Create `configs/cloudsim.yaml`:

```yaml
time:
  interval_s: 60
  steps_per_run: 360          # 6 simulated hours per run
  warmup_steps: 30
cluster:
  hosts: 30                   # one host per possible VM
  host_pes: 1
  host_mips: 1000
  vm_pes: 1
  vm_mips: 1000
  vm_min: 1
  vm_max: 30
  initial_vms: 4
  vm_startup_delay_s: 30      # boot time before a new VM can serve
requests:
  mean_service_s: 0.04        # 40 ms of CPU per request -> mu = 25 req/s per VM
  length_dist: exponential
  request_kb: 8               # for the network-in metric
  capacity_scale: 1.0         # see "Run time" below
balancer: least_outstanding   # or round_robin
slo: {rt_ms: 250.0, delta: 0.05}
actions: {k_min: -7, k_max: 7}
cost: {price_per_vm_hour: 0.10}
power: {model: square, p_idle_w: 80.0, p_max_w: 400.0}
bridge: {host: 127.0.0.1, port: 5555, timeout_s: 120}
```

The numbers mirror the synthetic system on purpose: 1000 MIPS and a mean request length of 40 million instructions (MI) give a mean service time of 40 ms and $\mu=25$ requests per second per VM. Results from Phase 2 and Phase 3 are then directly comparable.

**One VM per host.** Each host has exactly one processing element, so it can hold one VM. A host without a VM counts as powered off. This keeps "pool size" and "machines drawing power" the same thing, which the energy objective assumes. A consolidation scenario (several VMs per host) is a possible extension, not part of the plan.

**Run time.** A 6-hour run at a mean of about 200 requests per second is roughly 4.3 million requests, each a CloudSim Plus *cloudlet*. Measure the run time in Step 3.13. If one run takes longer than about ten minutes, set `capacity_scale` $s<1$: multiply every arrival rate by $s$ and divide the mean request length by $s$ (each request takes $1/s$ times longer). Utilisation $\lambda/(n\mu)$ is unchanged, so the pool sizes and the shape of every curve stay the same, while the number of cloudlets falls by the factor $s$. Response times scale by $1/s$, so the SLO must be scaled by $1/s$ too. With $s=0.25$: requests of 160 MI, SLO 1000 ms. Record $s$ in every result.

## Step 3.2: Java classes

All classes live in `cloudsim/src/main/java/org/uqautoscaler/sim/`. CloudSim Plus 8.x classes are under `org.cloudsimplus.*` (for example `org.cloudsimplus.core.CloudSimPlus`, `org.cloudsimplus.vms.VmSimple`, `org.cloudsimplus.brokers.DatacenterBrokerSimple`); let the IDE resolve imports and check names against the 8.5.7 Javadoc at `javadoc.io/doc/org.cloudsimplus/cloudsimplus/8.5.7`.

| Class | Responsibility |
|---|---|
| `Main` | parse arguments (`--config`, `--workload-csv`, `--seed`, `--policy`, `--out`, `--port`, `--force-action STEP:K`), build everything, start the simulation |
| `SimConfig` | read `cloudsim.yaml` with Jackson's YAML module, or have Python pass a JSON copy (simpler; add `jackson-databind` only) |
| `Cluster` | create the `CloudSimPlus` engine, `DatacenterSimple` with 30 `HostSimple`s, a `DatacenterBrokerSimple`, and the initial VMs |
| `ArrivalGenerator` | turn the per-interval rate series into Poisson arrival times and cloudlets |
| `LoadBalancer` | the broker's VM mapper: choose a VM for each arriving request |
| `ScalingActuator` | apply an action $k$: create VMs with a start-up delay, or drain and destroy VMs |
| `Telemetry` | per-interval metrics, response times and utilisation |
| `EnergyMeter` | per-interval energy from host utilisation and the power model |
| `ControlLoop` | clock-tick listener that closes each interval, logs it, asks the policy, and acts |
| `Policy` (interface) with `RandomTargetPolicy`, `CpuThresholdPolicy`, `RtThresholdPolicy`, `HoldPolicy`, `RemotePolicy` | decision rules; `RemotePolicy` asks Python |
| `RunLogger` | write the run log (CSV) and summary (JSON) |

Turn off CloudSim Plus's per-event console logging at the start of `Main` (its `Log` utility class has a `setLevel` method; use the "off" or "warn" level). Logging every event slows a run with millions of cloudlets dramatically.

## Step 3.3: Building the cluster

1. `CloudSimPlus sim = new CloudSimPlus(0.001);` The argument is the minimum time between events in seconds. Requests here last tens of milliseconds, so the engine must resolve at least 1 ms; the queueing test in Step 3.13 confirms the resolution is fine enough.
2. Create 30 hosts, each with one `PeSimple(1000)` and generous RAM, bandwidth and storage, and a `DatacenterSimple(sim, hostList)`.
3. `DatacenterBrokerSimple broker = new DatacenterBrokerSimple(sim);` Configure it so it neither shuts down nor destroys VMs on its own:
    * `broker.setShutdownWhenIdle(false)`: there are quiet moments between arrival batches;
    * `broker.setVmDestructionDelayFunction(null)`: per the 8.5.7 Javadoc, `null` disables automatic destruction of idle VMs; the autoscaler alone decides when VMs go away.
4. Each VM is a `VmSimple(1000, 1)` with a `CloudletSchedulerSpaceShared` scheduler: one request is served at a time and the rest wait in a FIFO queue, which is how a single-worker web server queues requests.
5. `sim.terminateAt(steps_per_run * interval_s)` ends the run.

## Step 3.4: Workloads and arrivals

Both environments must see identical request-rate series. Python generates them; Java reads them.

`scripts/p3_01_export_workloads.py` writes `data/raw/workloads/<shape>_seed<seed>.csv` (columns `step,rate`) for every shape and seed in `experiments.yaml`, using `make_workload` from Step 2.1 with `steps_per_run` from `cloudsim.yaml`.

`ArrivalGenerator`, at the start of each interval $t$ with rate $\lambda_t\cdot s$:

1. Draw arrival times as a Poisson process: inter-arrival gaps $\sim\mathrm{Exponential}(\lambda_t s)$ until the interval ends. Use one `java.util.SplittableRandom` seeded from `--seed` for arrivals and a second, separately seeded one for request lengths.
2. For each arrival create a `CloudletSimple(length, 1)` with $\text{length}=\max\big(1,\mathrm{round}(X)\big)$, $X\sim\mathrm{Exponential}$ with mean $\text{mean\_service\_s}\cdot\text{vm\_mips}/s$. Give it `UtilizationModelFull` for CPU and set its submission delay to (arrival time $-$ now). Store the planned arrival time in a `Map<Long, Double>` keyed by cloudlet id.
3. Submit the interval's cloudlets with `broker.submitCloudletList(list)`.

Attach a finish listener to every cloudlet (`addOnFinishListener`) that computes the response time as `finishTime - plannedArrival` and hands it to `Telemetry`.

**Important check (unit test `ArrivalTimingTest`).** The load balancer must choose a VM at the moment a request *arrives*, not when the batch is submitted at the start of the interval. Log `sim.clock()` inside the VM mapper for a few cloudlets with known delays and assert it equals their planned arrival times. If CloudSim Plus maps at submission time instead, change `ArrivalGenerator` to submit in one-second batches from the clock listener, so mapping happens at most one second early.

## Step 3.5: Load balancing

`LoadBalancer` is installed with `broker.setVmMapper(cloudlet -> ...)`. It considers only VMs that are created, have finished starting up (`!vm.isStartingUp()`) and are not draining, and picks the one with the fewest outstanding requests (keep a counter per VM: +1 when mapped, $-1$ when finished). Ties go to the lowest VM id. Return `Vm.NULL` only if no VM is eligible, and count those requests as dropped (they should never occur because `vm_min` is at least 1). The `round_robin` option exists as a sensitivity check.

## Step 3.6: Scaling actions

`ScalingActuator.apply(k)`:

* **Scale out** ($k>0$): create $k$ new `VmSimple`s, call `vm.setStartupDelay(vm_startup_delay_s)`, and `broker.submitVm(vm)`. They are billed and draw idle power from now on but receive no requests until start-up completes.
* **Scale in** ($k<0$): pick $|k|$ VMs, booting ones first, then the ones with the fewest outstanding requests. Mark them *draining*: the balancer stops sending them requests. When a draining VM has zero outstanding requests, call `broker.requestIdleVmDestruction(vm)`. A draining VM is billed until it is destroyed.
* Clamp so running plus booting VMs stay within `[vm_min, vm_max]`.

**Unit test `ScalingTest`.** Apply $+3$ then $-2$ with no load; after start-up delay and draining, the VM count matches; a booting VM receives no cloudlets; a draining VM finishes the requests it already has.

## Step 3.7: Telemetry per interval

`Telemetry` keeps running sums and, when an interval closes, produces these values:

| Metric | How it is measured |
|---|---|
| `rate` | arrivals in the interval / interval length |
| `rt_mean_ms`, `rt_p95_ms` | over requests that finished in the interval |
| `cpu_util` | busy time of serving VMs in the interval / (interval × serving VMs) × 100. Busy time comes from each finished cloudlet's start and finish times (`getStartTime()`, `getFinishTime()`), split across interval boundaries; add the busy part of still-running cloudlets at the boundary |
| `queue_len` | time-average of outstanding requests per serving VM, sampled at every clock tick |
| `net_in_kbps` | arrivals × `request_kb` × 8 / interval length |
| `throughput_rps` | completions / interval length |
| `mem_util` | $30+45\cdot\mathrm{cpu\_util}/100$ plus small Gaussian noise. CloudSim Plus does not model application memory pressure, so this is a modelling assumption; write it down. The metric filters decide whether it is used |
| `n_running`, `n_booting`, `n_draining` | pool state at the interval end |

The SLO violation flag is `rt_mean_ms > slo.rt_ms` (Step 1.3).

## Step 3.8: Energy and cost

`EnergyMeter` computes energy itself rather than relying on CloudSim Plus's host power accounting, so the power-model shape (linear, square, cubic, square root) is a configuration choice:

$$E_t=\sum_{\text{hosts with a VM}}\Big[P_{idle}+(P_{max}-P_{idle})\,h(u_{h,t})\Big]\cdot\frac{\Delta t}{3600}\ \text{Wh},$$

where $u_{h,t}$ is the busy fraction of the host's VM in interval $t$ (0 while booting). Cost is $c_t=(\text{running}+\text{booting}+\text{draining})\cdot p\cdot\Delta t/3600$.

As a cross-check, attach CloudSim Plus's linear host power model with the same idle and peak values in one test run and confirm its total energy matches `EnergyMeter` with `model: linear` to within 1%. If they differ, find out why before trusting either.

## Step 3.9: The control loop

`ControlLoop` registers `sim.addOnClockTickListener(info -> ...)`. On each tick it samples the outstanding-request counts (for `queue_len`), then checks whether the clock has reached the next interval boundary. At a boundary it:

1. closes the interval in `Telemetry` and `EnergyMeter`, and appends a row to the run log;
2. builds the observation (metrics, `n_running`, `n_booting`, `rt_mean_ms`, step index);
3. asks the policy for $k$ (or applies `--force-action` if this step matches);
4. calls `ScalingActuator.apply(k)`;
5. asks `ArrivalGenerator` for the next interval's cloudlets.

The first interval's arrivals are generated before `sim.start()`. Ticks happen only when events occur; with hundreds of requests per second there is an event at least every few milliseconds, so boundaries are hit with negligible delay. Record the actual boundary time in the log anyway.

## Step 3.10: Java-side policies

| Policy | Purpose |
|---|---|
| `HoldPolicy` | always $k=0$; used by tests |
| `RandomTargetPolicy` | calibration data collection (Step 3.14) |
| `CpuThresholdPolicy` | baseline; same rule and parameters as the Python version |
| `RtThresholdPolicy` | baseline; same rule and parameters as the Python version |
| `RemotePolicy` | sends the observation to Python and applies the returned $k$ |

Implementing the two threshold baselines in Java lets them run without the bridge. A test checks they make the same decisions as their Python twins on a recorded observation sequence.

## Step 3.11: The bridge protocol

Newline-delimited JSON over one TCP connection to `127.0.0.1:5555`. Java is the client. Messages:

```text
Java -> Python  {"type":"hello","run_id":"...","policy":"proposed","seed":3,
                 "config":{...cloudsim.yaml as JSON...}}
Python -> Java  {"type":"ready"}
Java -> Python  {"type":"observe","step":41,"time_s":2460.0,"n_running":9,
                 "n_booting":1,"metrics":{"cpu_util":71.2,...},"rt_mean_ms":183.4}
Python -> Java  {"type":"action","k":2}
...
Java -> Python  {"type":"bye","summary":{...}}
```

`src/autoscaler/bridge/protocol.py` defines these messages and validation. `src/autoscaler/bridge/server.py` accepts one connection, loads the models named in the hello message's policy, answers every `observe` with an `action`, writes the decision log, and exits after `bye`. If the server does not answer within `timeout_s`, Java aborts the run with a clear error instead of hanging.

## Step 3.12: Orchestration

`scripts/p3_02_run_cloudsim.py --policy proposed --workload step --seed 3 [--model-dir models/cloudsim]`:

1. builds the jar if it is older than the Java sources (`mvn -q -f cloudsim/pom.xml package`);
2. creates the run directory;
3. for remote policies, starts the bridge server in a background thread on a free port;
4. runs `java -Xmx4g -jar cloudsim/target/uq-autoscaler-sim-0.1.0.jar ...` with `subprocess.run`, passing the workload CSV, seed, port and output folder;
5. checks the exit code and that the run log has `steps_per_run` rows.

Java threshold policies run the same way without step 3.

## Step 3.13: Validation of the simulator

Write these as JUnit tests (`mvn test`) or Python integration tests (`pytest -m cloudsim`):

1. **Determinism.** Two runs with the same seed, workload and policy produce byte-identical run logs.
2. **Queueing sanity.** One VM, constant $\lambda=15$ req/s, $\mu=25$: the mean response time over a long run is within 5% of the M/M/1 value $1/(\mu-\lambda)=100$ ms.
3. **Utilisation.** Measured `cpu_util` is within 2 percentage points of $100\lambda/(n\mu)$ under constant load and a fixed pool.
4. **Start-up delay.** After $k=+1$, the new VM serves its first request no earlier than `vm_startup_delay_s` later.
5. **Arrival timing** (Step 3.4) and **scaling** (Step 3.6).
6. **Energy cross-check** (Step 3.8).
7. **Speed.** Time one 360-step `HoldPolicy` run on `daily_ramp_burst`; choose `capacity_scale` from it.

## Step 3.14: Collect calibration data inside CloudSim Plus

The models from Phase 2 learned the synthetic formula. They must be retrained on simulator data before CloudSim Plus experiments.

`RandomTargetPolicy` produces informative transitions: at each decision it draws a target utilisation $u^*\sim\mathrm{Uniform}(0.3,1.1)$, computes the pool $n^*=\lceil\lambda_{last}/(u^*\mu)\rceil$ that would reach it at the last interval's rate, sets $k=\mathrm{clip}(n^*-w,-7,7)$, and holds the resulting pool for two intervals. Targets above 1 deliberately visit overload, which the models must learn.

Run it on the `random_walk` and `daily_ramp_burst` shapes with the calibration seeds 1001 to 1010 (20 runs). `scripts/p3_03_extract_transitions.py` turns the run logs into the transition table of Step 1.3: for an action taken at the end of interval $t$, the before-metrics are interval $t$ and the after-metrics and target are interval $t+2$, the first interval in which booted VMs serve for the whole interval. Rows whose rate changed by more than 15% between $t$ and $t+2$ are kept, with both rates recorded, because the real controller faces the same drift. Write `data/processed/cloudsim_transitions.parquet`.

## Step 3.15: Retrain and validate on simulator data

Run the Phase 2 scripts with `--source cloudsim --model-dir models/cloudsim`:

1. metric filters and stage 1 (`p2_02`), with **grouped** cross-validation by `run_id` (`sklearn.model_selection.GroupKFold`);
2. hyperparameter search and ensemble training (`p2_03`), with train, calibration and test splits made by `group_split` on `run_id`;
3. calibration report (`p2_04`), end to end.

**Acceptance check 3.15.** End-to-end coverage at the 0.95 level is within 0.93 to 0.97 (with conformal correction if needed); the stage-1 $R^2$ table and the selected metrics are saved; the CloudSim Plus models live in `models/cloudsim/` and never overwrite the synthetic ones.

## Step 3.16: Counterfactual check of the predictor

A simulator can show what would have happened under a different action, which the base paper could not measure on a real cloud. `scripts/p3_04_counterfactual.py` picks 50 random (seed, step) pairs from held-out `HoldPolicy` runs and, for each $k$ in $\{-3,\dots,3\}$, re-runs the simulation from the start with the same seed and `--force-action STEP:k`. It compares the measured response time in interval STEP+2 with the model's $e^{\bar m_k}$ and $U_k$, and reports prediction error by $k$ and the coverage of $U_k$. This tests the models on exactly the counterfactual question the controller asks.

**Acceptance check 3.** Every simulator test passes; the counterfactual coverage of $U_k$ is within 0.90 to 0.98 for every $k$; one `proposed` run completes through the bridge and its decision log contains every candidate for every decision. Tag `git tag cloudsim-complete`.

# Phase 4: Experiments

## Step 4.1: Policies and the weight sweep

The four main policies are `cpu_threshold`, `rt_threshold`, `opt` and `proposed`. `proposed` runs with seven weight vectors $(w_L,w_C,w_E)$:

$$(1,1,1),\ (3,1,1),\ (1,3,1),\ (1,1,3),\ (1,3,3),\ (3,1,3),\ (3,3,1).$$

Each vector gives one point in (violation rate, cost, energy) space. Together they show the trade-offs the policy can reach, instead of one hand-picked setting. $(1,1,1)$ is the default used by the ablations.

## Step 4.2: Ablations

Each ablation changes exactly one thing relative to `proposed` with weights $(1,1,1)$:

| Name | Change | Question it answers |
|---|---|---|
| `abl_stage1_linear` | stage 1 = `paper_linear` | does non-linear metric estimation help? |
| `abl_single_member` | $M=1$ | what does the ensemble add? |
| `abl_no_uncertainty` | `mode="none"` ($U_k=e^{\bar m_k}$) | what does the chance constraint add? |
| `abl_epistemic_only` | `mode="epistemic"` | is member disagreement enough? |
| `abl_no_propagation` | `mode="ale_epi"` | what does stage-1 propagation add? |
| `abl_quantile_nn` | bound from `QuantileNet` | ensemble variance vs direct quantile |
| `abl_quantile_hgb` | bound from `QuantileHGB` | same, non-neural |
| `abl_min_cost` | selection rule `min_cost` | does the Pareto layer change outcomes? |
| `abl_two_objective` | objectives (cost, energy) only | does latency as an objective matter? |
| `abl_with_rate` | request rate as a stage-2 input (separately trained ensemble) | does the request rate help, as the base paper asked? |

## Step 4.3: Power-model sensitivity

Run `proposed` and `opt` on `daily_ramp_burst` with power models `linear`, `square` and `cubic` at $\Delta P/P_{idle}=4$, and `square` at $\Delta P/P_{idle}\in\{1, 1.42, 2\}$ (keep $P_{max}=400$ W and set $P_{idle}=400/(1+r)$). The analysis checks the threshold from the notes: the cost–energy trade-off appears only when $\Delta P/P_{idle}>1/u_{slo}^2\approx1.42$.

For policies whose decisions do not depend on energy (`cpu_threshold`, `rt_threshold`, `opt`), energy under another power model can be recomputed from the logged utilisation without re-running. `proposed` must be re-run because energy changes its decisions.

## Step 4.4: The grid file and the runner

Extend `configs/experiments.yaml`:

```yaml
weights_sweep: [[1,1,1],[3,1,1],[1,3,1],[1,1,3],[1,3,3],[3,1,3],[3,3,1]]
ablations: [abl_stage1_linear, abl_single_member, abl_no_uncertainty,
            abl_epistemic_only, abl_no_propagation, abl_quantile_nn,
            abl_quantile_hgb, abl_min_cost, abl_two_objective, abl_with_rate]
ablation_workloads: [daily_ramp_burst, flash_crowd]
sensitivity:
  workload: daily_ramp_burst
  policies: [proposed, opt]
  settings:
    - {model: linear, ratio: 4}
    - {model: square, ratio: 4}
    - {model: cubic,  ratio: 4}
    - {model: square, ratio: 1}
    - {model: square, ratio: 1.42}
    - {model: square, ratio: 2}
```

`scripts/p4_01_run_grid.py --env synthetic|cloudsim --part main|ablations|sensitivity --workers 4`:

* expands the grid into a list of runs, each with a unique `run_id` built from its settings (for example `cloudsim__proposed_w111__flash_crowd__s3`);
* skips runs whose folder already contains a complete run log, so an interrupted grid resumes where it stopped;
* runs jobs in parallel; for CloudSim Plus each worker gets its own bridge port (5555 + worker index);
* appends one line per finished run to `results/runs/manifest.csv` (run id, settings, wall-clock time, status).

Approximate sizes with 4 workloads and 10 seeds: main grid $(3+7)\times4\times10=400$ runs; ablations $10\times2\times10=200$; sensitivity $2\times6\times10=120$. In the synthetic environment the whole grid takes minutes. In CloudSim Plus, multiply the measured time of one run (Step 3.13) by 720 and divide by the number of workers to plan the compute. Run the main grid first, then the ablations, then the sensitivity runs.

**Acceptance check 4.** `manifest.csv` lists every planned run with status `ok`; no run log has missing steps; a random sample of five runs re-run from scratch gives identical logs.

# Phase 5: Analysis and the report

## Step 5.1: Per-run summaries

`src/autoscaler/eval/metrics.py`, `summarise(run_log, decision_log, cfg) -> dict`, computed after the warm-up:

| Metric | Definition |
|---|---|
| violation rate | share of intervals with `slo_violated` |
| mean and p95 response time | over interval means; p95 with linear interpolation |
| VM-hours, cost | $\sum_t n_t\Delta t$ and $p\sum_t n_t\Delta t$ (running + booting + draining) |
| energy | $\sum_t E_t$ in kWh |
| churn | number of decisions with $k\neq0$, and mean $|k|$ |
| fallback rate | share of decisions with no feasible candidate |
| realised coverage | share of decisions where the measured response time two intervals later (one in the synthetic environment) is $\le U_{k^*}$ for the chosen action |
| mean margin | mean of $U_{k^*}-e^{\bar m_{k^*}}$ |

`scripts/p5_01_aggregate.py` writes `results/tables/run_summaries.csv`, one row per run.

## Step 5.2: Statistics

`src/autoscaler/eval/stats.py`:

* mean and 95% confidence interval per (environment, workload, policy, metric): $\bar x\pm t_{0.975,R-1}\,s/\sqrt R$ with `scipy.stats.t.ppf(0.975, R-1)`;
* paired comparisons of `proposed` (weights $(1,1,1)$) against each baseline and each ablation, per workload and metric, using the per-seed differences: `scipy.stats.wilcoxon(x, y)`; report the median paired difference and the p-value;
* Holm correction across the comparisons of one table (sort p-values, compare the $i$-th smallest with $\alpha/(m-i+1)$), so many tests do not produce false discoveries by chance;
* hypervolume of each policy's points in normalised (violation rate, cost, energy) space with reference point $(1.1,1.1,1.1)$ after min-max normalisation over all policies of that workload; with three objectives and at most seven points, compute it exactly by inclusion–exclusion or with a simple slicing algorithm, and test it against a hand-computed two-point case.

`scripts/p5_02_stats.py` writes `results/tables/main_results.csv`, `ablations.csv`, `sensitivity.csv`, `paired_tests.csv` and `hypervolume.csv`.

## Step 5.3: Figures

`src/autoscaler/eval/plots.py` and `scripts/p5_03_figures.py`, all saved as PNG at 200 dpi and as PDF:

1. Workload shapes.
2. Example time series for one seed per workload: request rate, pool size for each main policy, response time against the SLO.
3. Bar charts with 95% intervals: violation rate, cost and energy per policy per workload.
4. Trade-off scatter: cost against violation rate, colour for energy, one marker per policy; the seven `proposed` weight points joined.
5. Ablation chart: change in each metric relative to `proposed`, per ablation.
6. Calibration: reliability diagram (nominal vs observed coverage) for every uncertainty mode and quantile model, end to end; conditional coverage by predicted-response-time bin.
7. Uncertainty decomposition: mean $\sigma_{ale}$, $\sigma_{epi}$, $\sigma_{stage1}$ against predicted utilisation.
8. Counterfactual check: predicted vs measured response time by $k$.
9. Power sensitivity: share of decisions with a multi-point cost–energy front, against $\Delta P/P_{idle}$, with the 1.42 threshold marked.

## Step 5.4: The report

`scripts/p5_04_report.py` builds `results/reports/final_report.pdf` (Matplotlib for figures; Pandoc with a LaTeX engine, or ReportLab, for the document). Sections:

1. Setup: environments, scenario numbers, `capacity_scale`, seeds, software versions (`requirements.lock.txt`, Java and CloudSim Plus versions), git commit.
2. Model quality: stage-1 $R^2$ table and selected metrics; stage-2 accuracy; calibration tables and figures.
3. Main results: tables with confidence intervals, paired tests, trade-off figure, hypervolume.
4. Ablations.
5. Power-model sensitivity.
6. Counterfactual check.
7. Synthetic vs CloudSim Plus: where the two environments agree and disagree.
8. Limitations and threats to validity, written from the numbers (for example, realised coverage below nominal, churn, the memory-metric assumption, one VM per host).

Every number in the report is read from the tables produced in Steps 5.1 and 5.2, never typed by hand.

# Phase 6: Reproducibility and hand-in

1. `scripts/run_all.py` runs every phase in order with one command and stops at the first failing acceptance check.
2. `README.md` explains the project in a paragraph, lists requirements, and gives the exact commands to reproduce the report.
3. Re-run `pip freeze > requirements.lock.txt`; record `java -version` and `mvn -v` in `docs/environment.txt`.
4. Tag the final commit (`git tag v1.0`) and archive `results/tables`, `results/figures` and `results/reports` with it.
5. Contact the base paper's corresponding author (Pintye) for their data and code if a direct comparison on their workloads is wanted; the paper states the data are available on request.

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Activate.ps1 cannot be loaded` | run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `mvn` uses the wrong Java | `JAVA_HOME` points elsewhere; fix it and reopen the terminal |
| `UnsupportedClassVersionError` | the jar was built or run with a different JDK; use JDK 21 for both |
| CloudSim Plus run is very slow | per-event logging is on, or too many cloudlets; turn logging off and lower `capacity_scale` |
| `OutOfMemoryError` in Java | raise `-Xmx`, shorten runs, or lower `capacity_scale`; do not keep references to finished cloudlets in your own code |
| Queueing sanity test fails | event resolution too coarse (Step 3.3), lengths not exponential, or the balancer sends work to booting VMs |
| Coverage far below 0.95 | stage-1 residuals were in-sample instead of out-of-fold, propagation is off, or the test fed true after-metrics; check each, then use conformal correction |
| Bridge hangs | Python raised an exception; the server must log tracebacks to the run folder and Java must time out |
| Results change between identical runs | a global random generator is used somewhere, or a dict/set iteration order affects decisions; seed every stream explicitly |
| Ensemble members are nearly identical | bootstrap not applied, or the same seed used for every member |
