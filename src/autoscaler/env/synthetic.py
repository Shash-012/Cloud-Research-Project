"""Formula-based synthetic system (Step 2.2)."""
import numpy as np


class SyntheticSystem:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        s = cfg["system"]
        self.vm_min, self.vm_max = s["vm_min"], s["vm_max"]
        self.mu = s["service_rate_per_vm"]
        self.L0 = s["base_latency_ms"]
        self.boot_delay = s["boot_delay_steps"]
        self.cx = s["complexity"]
        self.rt_noise = s["rt_noise_frac"]
        self.u_max = s["u_max"]
        self.rt_floor = s["rt_floor_ms"]
        self.tel = cfg["telemetry"]
        self.n_running = s["initial_vms"]
        self.booting: list[int] = []      # step at which each booting VM becomes ready

    # ---- hidden complexity -------------------------------------------------
    def complexity(self, t) -> float:
        eta = self.rng.normal(0.0, self.cx["noise_std"])
        return max(self.cx["floor"],
                   1.0 + self.cx["amplitude"] * np.sin(2 * np.pi * t / self.cx["period_steps"]) + eta)

    # ---- telemetry ---------------------------------------------------------
    def utilisation(self, rate, n_serving, c) -> float:
        return float(np.clip(rate * c / (n_serving * self.mu), 0.0, self.u_max))

    def noiseless_rt(self, u) -> float:
        return max(self.rt_floor, self.L0 / (1.0 - u))

    def observe(self, rate: float, n_serving: int, t, c: float | None = None) -> dict:
        """One interval's telemetry. `c` lets two observations share a complexity draw."""
        if c is None:
            c = self.complexity(t)
        r, tel = self.rng, self.tel
        u = self.utilisation(rate, n_serving, c)
        rt = max(self.rt_floor, self.L0 / (1.0 - u) * (1.0 + r.normal(0.0, self.rt_noise)))
        return {
            "cpu_util": float(np.clip(100 * u + r.normal(0.0, tel["cpu_util"]["noise_std"]), 0, 100)),
            "mem_util": float(np.clip(tel["mem_util"]["offset"] + tel["mem_util"]["slope"] * u
                                      + r.normal(0.0, tel["mem_util"]["noise_std"]), 0, 100)),
            "net_in_kbps": max(0.0, tel["net_in_kbps"]["per_request"] * rate
                               + r.normal(0.0, tel["net_in_kbps"]["noise_std"])),
            "disk_iops": max(0.0, tel["disk_iops"]["per_request"] * rate / n_serving
                             + r.normal(0.0, tel["disk_iops"]["noise_std"])),
            "queue_len": max(0.0, u / (1.0 - u) + r.normal(0.0, tel["queue_len"]["noise_std"])),
            "rt_ms": rt,
            "u": u,
        }

    # ---- pool --------------------------------------------------------------
    @property
    def n_booting(self) -> int:
        return len(self.booting)

    @property
    def n_total(self) -> int:
        return self.n_running + len(self.booting)

    def advance(self, t: int) -> None:
        """Promote booting VMs whose ready step has arrived."""
        ready = [s for s in self.booting if s <= t]
        self.booting = [s for s in self.booting if s > t]
        self.n_running += len(ready)

    def apply_action(self, k: int, t: int) -> int:
        """Apply k at the end of step t; returns the action actually applied after clamping."""
        target = int(np.clip(self.n_total + k, self.vm_min, self.vm_max))
        k_eff = target - self.n_total
        if k_eff > 0:
            self.booting.extend([t + self.boot_delay] * k_eff)
        elif k_eff < 0:
            remove = -k_eff
            # scale-in removes booting VMs first (latest-ready first), then running ones
            self.booting.sort()
            while remove and self.booting:
                self.booting.pop()
                remove -= 1
            self.n_running -= remove
        return k_eff
