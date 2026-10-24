"""Parallel grid runner."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .run_experiment import run_one


def _job(args):
    app, wl, pol, seed, dur = args
    return run_one(app, wl, pol, seed, duration_s=dur)


def run_grid(apps, workloads, policies, seeds, duration_s=None, workers=None, progress=True) -> pd.DataFrame:
    jobs = [(a, w, p, s, duration_s) for a in apps for w in workloads for p in policies for s in seeds]
    # The current CLI script is intentionally top-level; Windows spawn would
    # re-run it inside each worker. Use a reliable serial default there.
    workers = workers or (1 if os.name == "nt" else max(1, (os.cpu_count() or 2) - 1))
    out = []
    if workers == 1:
        for i, j in enumerate(jobs):
            out.append(_job(j))
            if progress: print(f"\r  {i + 1}/{len(jobs)}", end="", flush=True)
    else:
        with ProcessPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(_job, jobs)):
                out.append(r)
                if progress: print(f"\r  {i + 1}/{len(jobs)}", end="", flush=True)
    if progress: print()
    return pd.DataFrame(out)
