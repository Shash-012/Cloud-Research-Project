"""Generate synthetic calibration transitions (Step 2.3)."""
import argparse
from pathlib import Path

import numpy as np

from autoscaler.config import load_config, save_config
from autoscaler.data.calibration import collect_synthetic
from autoscaler.data.dataset import save_transitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = collect_synthetic(cfg, np.random.default_rng(cfg["calibration"]["seed"]))
    path = out / "synthetic_transitions.parquet"
    save_transitions(df, path)
    save_config(cfg, out / "synthetic_transitions_config.yaml")

    print(f"rows: {len(df)}  missing values: {int(df.isna().sum().sum())}")
    print(f"k range: {df['k'].min()} .. {df['k'].max()}  distinct: {sorted(df['k'].unique())}")
    print(f"share rt_after_ms > {cfg['slo']['rt_ms']}: {(df['rt_after_ms'] > cfg['slo']['rt_ms']).mean():.3f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
