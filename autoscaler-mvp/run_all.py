"""Run the whole pipeline end to end: calibrate, train, simulate.

    python run_all.py

Each stage is also runnable on its own from scripts/. Stage 2 exits non-zero if
its success criteria are not met, which stops the pipeline before the control
loop runs on a predictor that is not good enough to trust.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    "scripts/01_generate_calibration.py",
    "scripts/02_train_and_validate.py",
    "scripts/03_run_simulation.py",
]


def main():
    for stage in STAGES:
        print(f"\n{'=' * 72}\n{stage}\n{'=' * 72}")
        result = subprocess.run([sys.executable, str(ROOT / stage)], cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(f"\n{stage} failed (exit {result.returncode}); stopping.")
    print("\nall stages complete. figures in results/figures/")


if __name__ == "__main__":
    main()
