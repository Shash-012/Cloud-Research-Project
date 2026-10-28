"""Experiment grid runner (Step 4.4): resumable, parallel, one bridge port per worker, manifest.csv.

python scripts/p4_01_run_grid.py --env synthetic|cloudsim --part main|ablations|sensitivity --workers 4
Smoke grid: --policies ... --workloads ... --seeds ... --steps N (shortened runs).
"""
import argparse
import copy
import json
import queue
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from autoscaler.config import load_config
from autoscaler.eval.grid import Manifest, expand, is_complete

_W = {}


def _init_synthetic(proto, exp, model_dir):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    _W.update(proto=proto, exp=exp, model_dir=model_dir, models=None, limits=threadpool_limits(1))


def _run_synthetic(job, folder, steps):
    from autoscaler.decision.policies import load_models
    from autoscaler.sim.loop import episode, save_logs
    cfg = copy.deepcopy(_W["proto"])
    cfg["time"]["steps_per_run"] = steps
    cfg["time"]["warmup_steps"] = min(cfg["time"]["warmup_steps"], steps // 4)
    power = job.power(cfg["power"]["p_max_w"])
    if power:
        cfg["power"] = power
    models = None
    if job.policy not in ("cpu_threshold", "rt_threshold"):
        if _W["models"] is None:
            _W["models"] = load_models(_W["model_dir"])
        models = _W["models"]
    t0 = time.perf_counter()
    log, dec = episode(job.policy, job.workload, job.seed, cfg, _W["exp"], models)
    log["run_id"] = job.run_id
    save_logs(log, dec, folder)
    wall = time.perf_counter() - t0
    (Path(folder) / "done.json").write_text(json.dumps({"status": "ok", "wall_clock_s": wall, "steps": steps}))
    return "ok", wall


def _run_cloudsim(job, folder, steps, proto, cs, exp, model_dir, port):
    from autoscaler.sim.cloudsim import run_cloudsim
    cs = copy.deepcopy(cs)
    power = job.power(cs["power"]["p_max_w"])
    if power:
        cs["power"] = power
    if steps < cs["time"]["steps_per_run"]:
        cs["time"]["warmup_steps"] = min(cs["time"]["warmup_steps"], steps // 4)
    r = run_cloudsim(job.policy, job.workload, job.seed, folder, proto, cs, exp, model_dir, steps=steps,
                     run_id=job.run_id, port=port)
    (Path(folder) / "done.json").write_text(json.dumps({"status": r["status"], "wall_clock_s": r["wall_clock_s"],
                                                        "steps": steps, "port": r["port"]}))
    return r["status"], r["wall_clock_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["synthetic", "cloudsim"], required=True)
    ap.add_argument("--part", choices=["main", "ablations", "sensitivity"], required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--config", default=None, help="default: configs/prototype.yaml or configs/cloudsim.yaml")
    ap.add_argument("--prototype", default="configs/prototype.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--model-dir", default=None, help="default: models (synthetic) or models/cloudsim")
    ap.add_argument("--out", default=None, help="grid folder (default from experiments.yaml grid.out)")
    ap.add_argument("--manifest", default="results/runs/manifest.csv")
    ap.add_argument("--policies", nargs="*", default=None)
    ap.add_argument("--workloads", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="shorten runs (smoke grid)")
    args = ap.parse_args()

    proto, exp = load_config(args.prototype), load_config(args.experiments)
    cs = load_config(args.config or "configs/cloudsim.yaml") if args.env == "cloudsim" else None
    env_cfg = cs if cs else load_config(args.config or args.prototype)
    steps = args.steps or env_cfg["time"]["steps_per_run"]
    model_dir = args.model_dir or ("models/cloudsim" if args.env == "cloudsim" else "models")
    out = Path(args.out or exp["grid"]["out"]) / args.env
    workers = args.workers or exp["grid"]["workers"]
    manifest = Manifest(args.manifest)

    jobs = expand(exp, args.env, args.part, args.policies, args.workloads, args.seeds)
    todo = [j for j in jobs if not is_complete(out / j.run_id, steps)]
    print(f"{args.env}/{args.part}: {len(jobs)} runs planned, {len(jobs) - len(todo)} already complete, "
          f"{len(todo)} to run on {workers} workers", flush=True)

    t0, n_ok = time.perf_counter(), 0
    if args.env == "synthetic":
        pool = ProcessPoolExecutor(workers, initializer=_init_synthetic, initargs=(env_cfg, exp, model_dir))
        futs = {pool.submit(_run_synthetic, j, out / j.run_id, steps): j for j in todo}
    else:
        ports = queue.Queue()
        for i in range(workers):
            ports.put(exp["grid"]["base_port"] + i)        # one bridge port per worker

        def with_port(job):
            port = ports.get()
            try:
                return _run_cloudsim(job, out / job.run_id, steps, proto, cs, exp, model_dir, port)
            finally:
                ports.put(port)

        pool = ThreadPoolExecutor(workers)
        futs = {pool.submit(with_port, j): j for j in todo}
    with pool:
        for i, f in enumerate(as_completed(futs), 1):
            job = futs[f]
            try:
                status, wall = f.result()
            except Exception as exc:                      # record the failure; the grid can be resumed
                status, wall = f"error: {exc!r}", float("nan")
            n_ok += status == "ok"
            manifest.append(job, steps, wall, status, out / job.run_id)
            print(f"  [{i}/{len(todo)}] {job.run_id}: {status} ({wall:.1f}s)", flush=True)
    print(f"done: {n_ok}/{len(todo)} ok in {time.perf_counter() - t0:.0f}s; manifest {args.manifest}")
    if n_ok != len(todo):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
