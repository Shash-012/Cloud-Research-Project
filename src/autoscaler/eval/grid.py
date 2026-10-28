"""Experiment grid (Steps 4.1 to 4.4): expansion into runs with unique ids, completion checks, manifest."""
import csv
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    env: str
    part: str
    policy: str                  # name understood by make_policy (e.g. proposed_w311) or a Java policy
    workload: str
    seed: int
    power_model: str | None = None
    power_ratio: float | None = None

    @property
    def run_id(self) -> str:
        rid = f"{self.env}__{self.policy}__{self.workload}__s{self.seed}"
        if self.power_model is not None:
            rid += f"__pw_{self.power_model}_r{self.power_ratio:g}"
        return rid

    def power(self, p_max: float) -> dict | None:
        if self.power_model is None:
            return None
        return {"model": self.power_model, "p_idle_w": p_max / (1.0 + self.power_ratio), "p_max_w": p_max}


def weights_name(w) -> str:
    return "proposed_w" + "".join(str(int(x)) for x in w)


def expand(exp: dict, env: str, part: str, policies=None, workloads=None, seeds=None) -> list[Job]:
    """All runs of one part of the grid; the optional filters select a subset (e.g. a smoke grid)."""
    seeds = list(seeds or exp["eval_seeds"])
    jobs = []
    if part == "main":
        pols = list(exp["main_policies"]) + [weights_name(w) for w in exp["weights_sweep"]]
        wls = list(exp["workloads"])
        jobs = [Job(env, part, p, w, s) for p in pols for w in wls for s in seeds]
    elif part == "ablations":
        jobs = [Job(env, part, p, w, s) for p in exp["ablations"] for w in exp["ablation_workloads"] for s in seeds]
    elif part == "sensitivity":
        sens = exp["sensitivity"]
        jobs = [Job(env, part, p, sens["workload"], s, st["model"], float(st["ratio"]))
                for p in sens["policies"] for st in sens["settings"] for s in seeds]
    else:
        raise ValueError(f"unknown grid part {part}")
    if policies:
        jobs = [j for j in jobs if j.policy in policies]
    if workloads:
        jobs = [j for j in jobs if j.workload in workloads]
    ids = [j.run_id for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run ids in the grid")
    return jobs


def is_complete(folder, steps: int) -> bool:
    """A run is complete if its done.json says ok and its run log has every step."""
    folder = Path(folder)
    done = folder / "done.json"
    log = folder / "run_log.parquet"
    if not (done.exists() and log.exists()):
        return False
    if json.loads(done.read_text()).get("status") != "ok":
        return False
    import pandas as pd
    return len(pd.read_parquet(log, columns=["step"])) == steps


MANIFEST_FIELDS = ["run_id", "env", "part", "policy", "workload", "seed", "power_model", "power_ratio", "steps",
                   "wall_clock_s", "status", "run_dir"]


class Manifest:
    """results/runs/manifest.csv: one appended line per finished run (thread-safe)."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(MANIFEST_FIELDS)

    def append(self, job: Job, steps: int, wall: float, status: str, run_dir) -> None:
        row = {**asdict(job), "run_id": job.run_id, "steps": steps, "wall_clock_s": round(wall, 2),
               "status": status, "run_dir": str(run_dir)}
        with self.lock, open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k, "") for k in MANIFEST_FIELDS])
