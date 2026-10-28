"""Loading and saving transition tables."""
from pathlib import Path

import pandas as pd

from ..config import ROOT
from ..schema import transition_columns

SYNTHETIC_TRANSITIONS = ROOT / "data" / "processed" / "synthetic_transitions.parquet"
CLOUDSIM_TRANSITIONS = ROOT / "data" / "processed" / "cloudsim_transitions.parquet"


def save_transitions(df: pd.DataFrame, path, source: str = "synthetic") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    extra = [c for c in df.columns if c not in transition_columns(source)]
    df[transition_columns(source) + extra].to_parquet(path, index=False)


def load_transitions(path=SYNTHETIC_TRANSITIONS, source: str = "synthetic") -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(transition_columns(source)) - set(df.columns)
    if missing:
        raise ValueError(f"transition table lacks columns: {sorted(missing)}")
    return df
