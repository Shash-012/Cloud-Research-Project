"""Acceptance check 4, last item: re-run a random sample of finished grid runs from scratch and compare the logs.

python scripts/p4_02_rerun_check.py --n 5 --seed 4004 --max-wall-s 1800
--max-wall-s limits the sample to runs whose grid wall clock was at most that many seconds (the slowest CloudSim
runs took up to 17 h, which a re-run check cannot wait for); the limit is recorded in the table.
Writes results/tables/reproducibility.csv: for each sampled run, whether run_log.parquet (as a table), the
decoded decision log and, for CloudSim Plus, the Java run_log.csv (bytes) are identical to the grid's copy.
"""
import argparse
import gzip
import json
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from autoscaler.config import ROOT, load_config
from autoscaler.eval.grid import Job

spec = importlib.util.spec_from_file_location("p4_01_run_grid", ROOT / "scripts" / "p4_01_run_grid.py")
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


def job_of(row) -> Job:
    pm = None if pd.isna(row.power_model) or row.power_model == "" else str(row.power_model)
    pr = None if pm is None else float(row.power_ratio)
    return Job(row.env, row.part, row.policy, row.workload, int(row.seed), pm, pr)


def decisions_text(folder: Path) -> str | None:
    f = folder / "decisions.jsonl.gz"
    return gzip.open(f, "rt", encoding="utf-8").read() if f.exists() else None


def float_diffs(u, v, out: list) -> bool:
    """Collect relative differences of float fields; return False if any non-float field differs."""
    if isinstance(u, dict):
        return u.keys() == v.keys() and all([float_diffs(u[k], v[k], out) for k in u])
    if isinstance(u, list):
        return len(u) == len(v) and all([float_diffs(x, y, out) for x, y in zip(u, v)])
    if isinstance(u, float) and isinstance(v, float):
        out.append(abs(u - v) / max(abs(u), 1e-12))
        return True
    return u == v


def compare(orig: Path, new: Path) -> dict:
    a, b = pd.read_parquet(orig / "run_log.parquet"), pd.read_parquet(new / "run_log.parquet")
    da, db = decisions_text(orig), decisions_text(new)
    out = {"run_log_identical": bool(a.equals(b)), "steps": len(b), "decisions_identical": da == db}
    if da is not None and db is not None and da != db:
        # float32 network outputs can differ in the last bits between processes (thread scheduling in the BLAS
        # kernels); record whether every non-float field (actions, fallback, candidate set) is still equal
        rel, la, lb = [], da.splitlines(), db.splitlines()
        same = len(la) == len(lb) and all([float_diffs(json.loads(x), json.loads(y), rel) for x, y in zip(la, lb)])
        out["decisions_same_except_float_rounding"] = same
        out["decisions_max_rel_diff"] = max(rel) if rel else 0.0
    if (orig / "run_log.csv").exists():
        out["java_run_log_csv_identical"] = (orig / "run_log.csv").read_bytes() == (new / "run_log.csv").read_bytes()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/runs/manifest.csv")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=4004)
    ap.add_argument("--out", default="results/runs/rerun_check")
    ap.add_argument("--table", default="results/tables/reproducibility.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=5600, help="away from the grid's ports")
    ap.add_argument("--max-wall-s", type=float, default=None, help="sample only runs at most this long in the grid")
    ap.add_argument("--compare-only", action="store_true", help="re-compare existing re-runs in --out, no new runs")
    args = ap.parse_args()
    proto, exp = load_config(ROOT / "configs/prototype.yaml"), load_config(ROOT / "configs/experiments.yaml")
    cs = load_config(ROOT / "configs/cloudsim.yaml")

    man = pd.read_csv(args.manifest).drop_duplicates("run_id", keep="last")
    ok = man[man.status == "ok"]
    if args.max_wall_s is not None:
        ok = ok[ok.wall_clock_s <= args.max_wall_s]
    ok = ok.sort_values("run_id").reset_index(drop=True)
    sample = ok.iloc[np.random.default_rng(args.seed).choice(len(ok), size=args.n, replace=False)]
    print("re-running:", ", ".join(sample.run_id))
    if not args.compare_only:
        grid._init_synthetic(proto, exp, "models")

    def rerun(i_row):
        i, row = i_row
        job, new = job_of(row), Path(args.out) / row.run_id
        if args.compare_only:
            status, wall = ("ok" if (new / "run_log.parquet").exists() else "missing"), float("nan")
        elif job.env == "synthetic":
            status, wall = grid._run_synthetic(job, new, int(row.steps))
        else:
            status, wall = grid._run_cloudsim(job, new, int(row.steps), proto, cs, exp, "models/cloudsim",
                                              args.base_port + i)
        res = {"run_id": row.run_id, "env": row.env, "status": status, "rerun_wall_clock_s": round(wall, 1)}
        if status == "ok":
            res.update(compare(Path(row.run_dir), new))
        print(res, flush=True)
        return res

    with ThreadPoolExecutor(min(args.workers, args.n)) as pool:
        rows = list(pool.map(rerun, enumerate(r for _, r in sample.iterrows())))
    df = pd.DataFrame(rows)
    if args.compare_only and Path(args.table).exists():     # keep the wall clock of the earlier re-run
        prev = pd.read_csv(args.table).set_index("run_id")["rerun_wall_clock_s"]
        df["rerun_wall_clock_s"] = df.run_id.map(prev)
    df["sample_seed"] = args.seed
    df["max_wall_s"] = args.max_wall_s
    df.to_csv(args.table, index=False)
    cols = [c for c in df.columns if c.endswith("_identical")]
    print(f"{int(df[cols].fillna(True).all(axis=1).sum())}/{len(df)} runs identical -> {args.table}")
    if "decisions_same_except_float_rounding" in df:
        print("decision logs differing only in float rounding:", int(df.decisions_same_except_float_rounding.eq(True).sum()),
              "max relative difference", df.decisions_max_rel_diff.max())


if __name__ == "__main__":
    main()
