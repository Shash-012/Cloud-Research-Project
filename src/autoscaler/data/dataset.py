"""Loading and saving transition tables."""
from pathlib import Path

import pandas as pd

from ..config import ROOT
from ..schema import TRANSITION_COLUMNS

SYNTHETIC_TRANSITIONS = ROOT / "data" / "processed" / "synthetic_transitions.parquet"


def save_transitions(df: pd.DataFrame, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df[TRANSITION_COLUMNS].to_parquet(path, index=False)


def load_transitions(path=SYNTHETIC_TRANSITIONS) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(TRANSITION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"transition table lacks columns: {sorted(missing)}")
    return df
