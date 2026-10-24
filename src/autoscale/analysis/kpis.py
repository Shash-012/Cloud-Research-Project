"""Performance measures (paper Sec 7.1): QoS violation %, RTavg, VM used."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_kpis(intervals: pd.DataFrame, requests: pd.DataFrame, qos_ms: float, max_vms: int) -> dict:
    rt = requests["rt_ms"].to_numpy()
    viol = rt > qos_ms
    vm_used = float(intervals["w"].sum())
    return dict(
        n_requests=int(len(rt)),
        qos_viol_pct=100.0 * float(viol.mean()) if len(rt) else float("nan"),
        rt_avg_ms=float(rt.mean()) if len(rt) else float("nan"),
        rt_avg_viol_ms=float(rt[viol].mean()) if viol.any() else 0.0,   # paper: 'during violations'
        rt_p95_ms=float(np.quantile(rt, 0.95)) if len(rt) else float("nan"),
        vm_used=vm_used,
        vm_used_pct=100.0 * vm_used / (max_vms * len(intervals)),
        mean_vms=float(intervals["w"].mean()),
        std_vms=float(intervals["w"].std()),
        scale_events=int((intervals["delta"] != 0).sum()),
    )
