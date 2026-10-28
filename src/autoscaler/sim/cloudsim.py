"""Running the CloudSim Plus simulator from Python (Steps 3.4 and 3.12)."""
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from ..config import ROOT, java_config, merge_cloudsim
from ..env.workloads import make_workload
from ..seeds import STREAM_WORKLOAD, child_rngs

CLOUDSIM_DIR = ROOT / "cloudsim"
JAR = CLOUDSIM_DIR / "target" / "uq-autoscaler-sim-0.1.0.jar"
WORKLOAD_DIR = ROOT / "data" / "raw" / "workloads"
JAVA_POLICIES = ("hold", "random_target", "cpu_threshold", "rt_threshold")


def ensure_jar() -> Path:
    """Build the jar if it is missing or older than any Java source or the pom."""
    sources = list((CLOUDSIM_DIR / "src" / "main").rglob("*.java")) + [CLOUDSIM_DIR / "pom.xml"]
    newest = max(p.stat().st_mtime for p in sources)
    if not JAR.exists() or JAR.stat().st_mtime < newest:
        mvn = shutil.which("mvn") or shutil.which("mvn.cmd")      # mvn.cmd on Windows
        if mvn is None:
            raise FileNotFoundError("Maven (mvn) not found on PATH")
        subprocess.run([mvn, "-q", "-f", str(CLOUDSIM_DIR / "pom.xml"), "package", "-DskipTests"], check=True)
    return JAR


def workload_csv(shape: str, seed: int, steps: int, params: dict, folder=WORKLOAD_DIR) -> Path:
    """data/raw/workloads/<shape>_seed<seed>.csv (columns step,rate); same stream as the synthetic runs."""
    folder = Path(folder)
    path = folder / f"{shape}_seed{seed}.csv"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        rates = make_workload(shape, steps, child_rngs(seed, 3)[STREAM_WORKLOAD], params)
        pd.DataFrame({"step": range(steps), "rate": rates}).to_csv(path, index=False, float_format="%.6f")
    return path


def read_run_log(folder) -> pd.DataFrame:
    df = pd.read_csv(Path(folder) / "run_log.csv")
    for c in ("fallback", "slo_violated"):
        df[c] = df[c].astype(str).str.lower().eq("true")
    return df


def run_cloudsim(policy: str, workload: str, seed: int, out_dir, proto: dict, cs: dict, exp: dict,
                 model_dir=None, steps: int | None = None, force_action: tuple[int, int] | None = None,
                 initial_vms: int | None = None, run_id: str | None = None, energy_crosscheck: bool = False,
                 xmx: str = "4g", port: int = 0, remote: bool = False) -> dict:
    """Run one simulation; Python policies are served by a bridge on a free (or given) port.

    Returns a dict with status, wall-clock time and the run directory. The run log is also saved as Parquet.
    """
    from ..bridge.server import serve_run
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jar = ensure_jar()
    n_steps = steps or cs["time"]["steps_per_run"]
    wl = workload_csv(workload, seed, cs["time"]["steps_per_run"], proto["workloads"])
    run_id = run_id or f"cloudsim__{policy}__{workload}__s{seed}"

    bridge = None
    if policy not in JAVA_POLICIES or remote:
        cfg = merge_cloudsim(proto, cs, exp)
        bridge = serve_run(cfg, exp, model_dir, out_dir, policy, timeout_s=cs["bridge"]["timeout_s"], port=port)
        port = bridge.port
    jcfg = java_config(cs, exp, port if bridge else None)
    (out_dir / "sim_config.json").write_text(json.dumps(jcfg, indent=2))

    cmd = ["java", f"-Xmx{xmx}", "-jar", str(jar), "--config", str(out_dir / "sim_config.json"),
           "--workload-csv", str(wl), "--workload-name", workload, "--seed", str(seed), "--policy", policy,
           "--out", str(out_dir), "--run-id", run_id, "--steps", str(n_steps)]
    if bridge:
        cmd += ["--port", str(port), "--remote"]
    if force_action is not None:
        cmd += ["--force-action", f"{force_action[0]}:{force_action[1]}"]
    if initial_vms is not None:
        cmd += ["--initial-vms", str(initial_vms)]
    if energy_crosscheck:
        cmd += ["--energy-crosscheck"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    (out_dir / "java_stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if bridge:
        bridge.join(timeout=30)

    status = "ok"
    rows = 0
    if proc.returncode != 0:
        status = f"java exit {proc.returncode}: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}"
    elif bridge and bridge.error:
        status = f"bridge error: {bridge.error}"
    else:
        log = read_run_log(out_dir)
        rows = len(log)
        if rows != n_steps:
            status = f"run log has {rows} rows, expected {n_steps}"
        log.to_parquet(out_dir / "run_log.parquet", index=False)
    return {"run_id": run_id, "policy": policy, "workload": workload, "seed": seed, "status": status,
            "rows": rows, "wall_clock_s": wall, "run_dir": str(out_dir), "port": port if bridge else None}
