"""Base-paper metric selection filters (Step 2.4)."""
import json
from pathlib import Path

import pandas as pd
from scipy import stats


def ttest_filter(df: pd.DataFrame, metrics, alpha: float) -> dict:
    """Welch t-test of (after - before) between scale-up and scale-down rows.

    Returns {metric: {"p": p_value, "keep": p < alpha}}.
    """
    up, down = df[df["k"] > 0], df[df["k"] < 0]
    out = {}
    for m in metrics:
        a = up[f"{m}_after"] - up[f"{m}_before"]
        b = down[f"{m}_after"] - down[f"{m}_before"]
        p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        out[m] = {"p": p, "keep": bool(p < alpha)}
    return out


def r2_filter(scores: dict, threshold: float) -> dict:
    """scores: {metric: {candidate: cv_r2}}. Keep a metric if its best candidate exceeds threshold."""
    return {m: bool(max(s.values()) > threshold) for m, s in scores.items()}


def final_metric_set(ttest: dict, r2: dict, order) -> list[str]:
    return [m for m in order if ttest.get(m, {}).get("keep") and r2.get(m, False)]


def save_selected(metrics: list[str], path, details: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"selected": metrics, **(details or {})}, f, indent=2)


def load_selected(path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["selected"]
