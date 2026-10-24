"""Workload generators. All are precomputed per second from a seed, so runs are deterministic
and every policy sees the identical load for a given seed (paired comparison)."""
from __future__ import annotations
import numpy as np

UTIL_AT_PEAK = 0.7


class Workload:
    def __init__(self, name: str, lam: np.ndarray, cx: np.ndarray):
        self.name, self.lam, self.cx = name, lam, cx

    def rate_at(self, t: float) -> float:
        return float(self.lam[min(int(t), len(self.lam) - 1)])

    def complexity_at(self, t: float) -> float:
        return float(self.cx[min(int(t), len(self.cx) - 1)])


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    k = np.ones(win) / win
    return np.convolve(np.pad(x, (win, win), mode="edge"), k, mode="same")[win:-win]


def _triangle(n: int, period: float, lo: float, hi: float, phase: float = 0.0) -> np.ndarray:
    t = (np.arange(n) / period + phase) % 1.0
    tri = 1 - np.abs(2 * t - 1)        # 0..1..0
    return lo + (hi - lo) * tri


def build_workload(wl_cfg: dict, app_cfg: dict, duration_s: float, seed: int) -> Workload:
    n = int(duration_s) + 600          # slack for draining
    rng = np.random.default_rng(10_000 + seed)
    cap = app_cfg["capacity_per_vm"]
    shape = wl_cfg["shape"]
    cx = np.ones(n)
    if shape == "const_rr":
        rate = wl_cfg["const_vms"] * cap * UTIL_AT_PEAK
        lam = np.full(n, rate)
        lam[:120] = np.linspace(0.0, rate, 120)         # short ramp 0 -> rate (paper: 0 to 40 req/s)
        i = 0
        while i < n:
            span = int(rng.uniform(120, 360))
            cx[i:i + span] = rng.uniform(0.5, 2.0)
            i += span
        return Workload(shape, lam, cx)

    peak = wl_cfg["peak_vms"] * cap * UTIL_AT_PEAK
    if shape == "webshop":
        t = np.arange(n) / max(duration_s, 1)
        s = 0.5 + 0.38 * np.sin(2 * np.pi * (t * 1.0) - np.pi / 2)
        ou = _smooth(rng.normal(0, 0.08, n), 90)
        s = np.clip(s + ou, 0.12, 1.0)
    elif shape == "surge":
        levels = np.array([0.12, 0.25, 0.5, 0.75, 1.0])
        s = np.zeros(n); i = 0; prev = 0.3
        while i < n:
            span = int(rng.uniform(180, 420))
            lv = rng.choice(levels[levels != prev]); prev = lv
            s[i:i + span] = lv; i += span
    elif shape == "waves":
        s = _smooth(_triangle(n, 600, 0.12, 1.0, phase=0.1), 20)
    elif shape == "ramp":
        s = _triangle(n, wl_cfg.get("period_s", 1800), 0.3, 1.0)
    else:
        raise ValueError(f"unknown workload shape {shape}")
    return Workload(shape, np.maximum(s * peak, 0.2), cx)
