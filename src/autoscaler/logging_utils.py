"""Run directories: every script writes only inside its own one (Step 1.4)."""
import subprocess
from datetime import datetime
from pathlib import Path

from .config import ROOT, save_config


def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_dir(base, name: str, cfg: dict | None = None) -> Path:
    """Create <base>/<timestamp>_<name>/ with config.yaml and git_rev.txt; return it."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = Path(base) / f"{stamp}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    save_config(cfg or {}, path / "config.yaml")
    (path / "git_rev.txt").write_text(git_head() + "\n", encoding="utf-8")
    return path
