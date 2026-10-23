# Autoscaling Research: Part 1

This repository is the Part 1 reproduction scaffold for the base-paper
autoscaling study by Pintye, Kovács, and Lovas (2024). It uses a deterministic
Python/SimPy simulator rather than a real cloud or CloudSim Plus.

The simulator is hidden behind `autoscale.env.base.Environment`, so Part 2 can
replace it with a CloudSim Plus bridge without changing policies or analysis.
The implementation scope is intentionally limited to the base-paper policies:

- RT-threshold
- CPU-threshold
- reactive OPT
- proactive OPT

## Setup

Python 3.11 is the supported runtime. Create an environment and install the
package with development tools:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

The source implementation is not complete yet; the initial milestone is a
clean, testable architecture. See [docs/decisions.md](docs/decisions.md) for
the pinned choices needed for a faithful reproduction.

## Planned workflow

```text
simulator -> perturbation data -> metric selection -> RT model
     -> policy/orchestrator loop -> tidy experiment CSVs -> analysis
```

Part 2 CloudSim integration is deliberately out of scope for this branch.
