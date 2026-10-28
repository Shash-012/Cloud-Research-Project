"""Every policy and ablation on every workload and evaluation seed, synthetic environment (Step 2.13)."""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from autoscaler.config import load_config, save_config
from autoscaler.logging_utils import run_dir

_W = {}


def _init(cfg, exp, model_dir):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    _W["limits"] = threadpool_limits(1)
    _W.update(cfg=cfg, exp=exp, model_dir=model_dir, models=None)


def _models():
    from autoscaler.decision.policies import load_models
    if _W["models"] is None:
        _W["models"] = load_models(_W["model_dir"])
    return _W["models"]


def _job(policy, workload, seed, out_dir):
    from autoscaler.eval.metrics import summarise
    from autoscaler.sim.loop import episode, save_logs
    t0 = time.perf_counter()
    needs_models = policy not in ("cpu_threshold", "rt_threshold")
    log, dec = episode(policy, workload, seed, _W["cfg"], _W["exp"], _models() if needs_models else None)
    elapsed = time.perf_counter() - t0
    run_id = log.run_id.iloc[0]
    save_logs(log, dec, Path(out_dir) / run_id)
    return {"run_id": run_id, "env": "synthetic", "policy": policy, "workload": workload, "seed": seed,
            "wall_clock_s": elapsed, "status": "ok", **summarise(log, dec, _W["cfg"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--policies", nargs="*", default=None)
    ap.add_argument("--workloads", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--no-analysis", action="store_true")
    args = ap.parse_args()
    cfg, exp = load_config(args.config), load_config(args.experiments)
    rd = run_dir(args.out, "p2_06_sweep", cfg)
    save_config(exp, rd / "experiments.yaml")
    runs_dir = rd / "runs"

    policies = args.policies or (exp["policies"] + exp["ablations"])
    workloads = args.workloads or exp["workloads"]
    seeds = args.seeds or exp["eval_seeds"]
    jobs = [(p, w, s) for p in policies for w in workloads for s in seeds]
    workers = args.workers or exp["sweep"]["workers"]
    print(f"{len(jobs)} runs on {workers} workers -> {rd}", flush=True)

    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(cfg, exp, args.model_dir)) as pool:
        futs = {pool.submit(_job, p, w, s, str(runs_dir)): (p, w, s) for p, w, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            p, w, s = futs[f]
            try:
                rows.append(f.result())
            except Exception as exc:          # record and continue; the manifest shows the failure
                rows.append({"run_id": f"synthetic__{p}__{w}__s{s}", "policy": p, "workload": w, "seed": s,
                             "status": f"error: {exc!r}"})
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {time.perf_counter() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values(["policy", "workload", "seed"])
    df.to_csv(rd / "run_summaries.csv", index=False)
    df.reindex(columns=["run_id", "policy", "workload", "seed", "wall_clock_s", "status"]).to_csv(rd / "manifest.csv", index=False)
    (Path(args.out) / "latest_synthetic_sweep.txt").write_text(str(rd))
    bad = df[df.status != "ok"]
    print(f"finished: {len(df) - len(bad)} ok, {len(bad)} failed, {time.perf_counter() - t0:.0f}s")
    if len(bad):
        print(bad[["run_id", "status"]].to_string(index=False))
    if not args.no_analysis:
        subprocess.run([sys.executable, "scripts/p5_01_aggregate.py", "--env", "synthetic", "--sweep", str(rd),
                        "--config", args.config], check=True)


if __name__ == "__main__":
    main()
