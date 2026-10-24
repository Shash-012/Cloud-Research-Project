"""Tune threshold and OPT policy parameters on a held-out tuning seed."""

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import _common  # noqa: F401
import yaml

from autoscale.config import RESULTS, load_config
from autoscale.runner.run_experiment import run_one


def build_jobs(config: dict, args: argparse.Namespace) -> list[tuple]:
    rt_up = [0.6, 0.8] if args.fast else [0.5, 0.6, 0.7, 0.8, 0.9]
    rt_low = [0.25] if args.fast else [0.15, 0.25, 0.35]
    cpu_up = [55, 75] if args.fast else [45, 55, 65, 75, 85]
    cpu_low = [25] if args.fast else [15, 25, 35]
    opt_fraction = [0.5, 0.8] if args.fast else [0.4, 0.55, 0.7, 0.85, 1.0]
    jobs = []
    for app in args.apps or list(config["apps"]):
        for workload in args.workloads:
            for upper, lower in itertools.product(rt_up, rt_low):
                if lower < upper:
                    jobs.append((app, workload, "rt", {"upper_frac": upper, "lower_frac": lower}))
            for upper, lower in itertools.product(cpu_up, cpu_low):
                if lower < upper:
                    jobs.append((app, workload, "cpu", {"upper": upper, "lower": lower}))
            for fraction in opt_fraction:
                for policy in ("opt_reactive", "opt_proactive"):
                    if (RESULTS / "models" / app).exists():
                        jobs.append((app, workload, policy, {"target_frac": fraction}))
    return jobs


def run_job(job: tuple, config: dict, duration: float) -> tuple:
    app, workload, policy, params = job
    kpis = run_one(
        app,
        workload,
        policy,
        config["tuning_seed"],
        cfg=config,
        tuned={f"{app}/{workload}": {policy: params}},
        duration_s=duration,
    )
    return job, kpis["qos_viol_pct"] + 0.15 * kpis["vm_used_pct"]


def _run_job_args(args: tuple) -> tuple:
    """Process-pool adapter kept at module scope for Windows spawn."""
    return run_job(*args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--workloads", nargs="*", default=["webshop", "surge", "waves"])
    parser.add_argument("--apps", nargs="*")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    duration = args.duration or config["duration_s"]
    jobs = build_jobs(config, args)
    print(len(jobs), "tuning runs")
    workers = args.workers or min(4, max(1, (os.cpu_count() or 2) - 1))
    if workers == 1:
        results = []
        for index, job in enumerate(jobs, 1):
            results.append(run_job(job, config, duration))
            print(f"\r  {index}/{len(jobs)}", end="", flush=True)
        print()
    else:
        job_args = [(job, config, duration) for job in jobs]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = []
            for index, result in enumerate(executor.map(_run_job_args, job_args), 1):
                results.append(result)
                print(f"\r  {index}/{len(jobs)}", end="", flush=True)
        print()

    best = {}
    for (app, workload, policy, params), score in results:
        current = best.setdefault(f"{app}/{workload}", {}).get(policy)
        if current is None or score < current[0]:
            best[f"{app}/{workload}"][policy] = (score, params)
    output = {
        key: {policy: value[1] for policy, value in policies.items()}
        for key, policies in best.items()
    }
    destination = RESULTS / "tables" / "tuned_baselines.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(output), encoding="utf-8")
    print(yaml.safe_dump(output))


if __name__ == "__main__":
    main()
