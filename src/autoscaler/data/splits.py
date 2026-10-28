"""Train / calibration / test splits (Step 2.3)."""
import numpy as np
import pandas as pd


def _cuts(n: int, fracs) -> tuple[int, int]:
    f = np.asarray(fracs, dtype=float)
    f = f / f.sum()
    a = int(round(f[0] * n))
    b = int(round((f[0] + f[1]) * n))
    return a, b


def random_split(df: pd.DataFrame, fracs, seed: int):
    """Random row split into (train, calib, test). Valid for independent rows."""
    idx = np.random.default_rng(seed).permutation(len(df))
    a, b = _cuts(len(df), fracs)
    return (df.iloc[idx[:a]].reset_index(drop=True),
            df.iloc[idx[a:b]].reset_index(drop=True),
            df.iloc[idx[b:]].reset_index(drop=True))


def group_split(df: pd.DataFrame, group_col: str, fracs, seed: int):
    """Split whole groups (e.g. run_id) so correlated rows stay on one side."""
    groups = np.array(sorted(df[group_col].unique()))
    groups = groups[np.random.default_rng(seed).permutation(len(groups))]
    a, b = _cuts(len(groups), fracs)
    parts = (set(groups[:a]), set(groups[a:b]), set(groups[b:]))
    return tuple(df[df[group_col].isin(p)].reset_index(drop=True) for p in parts)


def split_from_config(df: pd.DataFrame, cfg: dict):
    s = cfg["splits"]
    return random_split(df, [s["train"], s["calib"], s["test"]], s["seed"])
