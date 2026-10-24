"""SimPy discrete-event cloud: load balancer -> per-VM queues -> VM pool, with boot delay.

Faithful to the paper's testbed simplifications: the load balancer dispatches evenly (round robin)
and has NO drop, NO timeout and NO retry, so overload shows up as runaway response time.
A request holds a worker-thread slot, uses a vCPU for its CPU phase and (optionally) a disk channel
for its I/O phase. CPU utilisation therefore under-reports load for disk-bound apps (db_io).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import simpy

from .base import Env, Observation
from .apps import AppProfile
from .metrics_gen import MetricGenerator
from .workloads import Workload, build_workload


class _Acc:
    __slots__ = ("cpu_time", "io_time", "arrivals", "completions", "kb_in", "kb_out", "iops",
                 "s_inflight", "s_wait", "s_runq", "n")

    def __init__(self):
        self.reset()

    def reset(self):
        self.cpu_time = self.io_time = self.kb_in = self.kb_out = 0.0
        self.arrivals = self.completions = self.iops = 0
        self.s_inflight = self.s_wait = self.s_runq = 0.0
        self.n = 0


class _VM:
    def __init__(self, sim: simpy.Environment, app: AppProfile, vid: int):
        self.vid = vid
        self.threads = simpy.Resource(sim, capacity=app.slots)
        self.cpu = simpy.Resource(sim, capacity=app.vcpu)
        self.io = simpy.Resource(sim, capacity=app.io_cap) if app.io_cap > 0 and app.io_ms > 0 else None
        self.acc = _Acc()

    def in_system(self) -> int:
        return self.threads.count + len(self.threads.queue)


class SimpyEnv(Env):
    def __init__(self, app_cfg: dict, workload: Workload, cfg: dict, seed: int, duration_s: float | None = None):
        self.app = AppProfile.from_cfg(app_cfg)
        self.wl = workload
        self.cfg = cfg
        self.seed = seed
        self.dt = float(cfg["interval_s"])
        self.boot = float(cfg["boot_delay_s"])
        self.min_vms, self.max_vms = int(cfg["min_vms"]), int(cfg["max_vms"])
        self.duration = float(duration_s if duration_s is not None else cfg["duration_s"])
        self.t = 0.0

    # ---------- lifecycle ----------
    def reset(self, init_vms: int | None = None) -> Observation:
        self.sim = simpy.Environment()
        self.rng = np.random.default_rng(self.seed + 17)
        self.gen = MetricGenerator(self.seed, self.dt)
        self.vms: list[_VM] = []
        self._vid = 0
        self._rr = 0
        self._pending: list[dict] = []
        self._arrivals_on = True
        self._inflight_total = 0
        self.t = 0.0
        self._rt_int: list[float] = []
        self._arr_int = 0
        self._req_t0: list[float] = []
        self._req_rt: list[float] = []
        self._req_w: list[int] = []
        self._rows: list[dict] = []
        n0 = int(init_vms if init_vms is not None else self.cfg["init_vms"])
        for _ in range(max(n0, self.min_vms)):
            self._attach()
        self.sim.process(self._arrivals())
        self.sim.process(self._sampler())
        return self.step(0)

    @property
    def done(self) -> bool:
        return self.t >= self.duration - 1e-9

    @property
    def w(self) -> int:
        return len(self.vms)

    @property
    def pending(self) -> int:
        return len(self._pending)

    # ---------- orchestrator ----------
    def _attach(self):
        self.vms.append(_VM(self.sim, self.app, self._vid)); self._vid += 1

    def _boot(self, token):
        yield self.sim.timeout(self.boot)
        if not token["cancelled"]:
            self._pending.remove(token)
            self._attach()

    def _apply(self, delta: int):
        total = self.w + self.pending
        target = int(np.clip(total + delta, self.min_vms, self.max_vms))
        diff = target - total
        if diff > 0:
            for _ in range(diff):
                tok = {"cancelled": False}
                self._pending.append(tok)
                if self.boot <= 0:
                    self._pending.remove(tok); self._attach()
                else:
                    self.sim.process(self._boot(tok))
        elif diff < 0:
            for _ in range(-diff):
                if self._pending:                      # cancel the newest booting VM first
                    self._pending.pop()["cancelled"] = True
                elif self.w > self.min_vms:            # then detach the least-loaded attached VM
                    vm = min(self.vms, key=lambda v: v.in_system())
                    self.vms.remove(vm)
                    self._rr = 0

    # ---------- processes ----------
    def _arrivals(self):
        while self._arrivals_on:
            lam = max(self.wl.rate_at(self.sim.now), 1e-3)
            yield self.sim.timeout(self.rng.exponential(1.0 / lam))
            if not self._arrivals_on:
                break
            cx = self.wl.complexity_at(self.sim.now)
            cpu_s, io_s, kin, kout = self.app.sample(self.rng, cx)
            self._rr = (self._rr + 1) % len(self.vms)
            vm = self.vms[self._rr]
            vm.acc.arrivals += 1; vm.acc.kb_in += kin
            self._arr_int += 1
            self._inflight_total += 1
            self.sim.process(self._request(vm, self.sim.now, cpu_s, io_s, kout, len(self.vms)))

    def _request(self, vm: _VM, t0: float, cpu_s: float, io_s: float, kout: float, w_at: int):
        with vm.threads.request() as slot:
            yield slot
            with vm.cpu.request() as c:
                yield c
                yield self.sim.timeout(cpu_s)
            vm.acc.cpu_time += cpu_s
            if io_s > 0 and vm.io is not None:
                with vm.io.request() as i:
                    yield i
                    yield self.sim.timeout(io_s)
                vm.acc.io_time += io_s; vm.acc.iops += 1
        rt = self.sim.now - t0
        vm.acc.completions += 1; vm.acc.kb_out += kout
        self._rt_int.append(rt * 1000.0)
        self._req_t0.append(t0); self._req_rt.append(rt * 1000.0); self._req_w.append(w_at)
        self._inflight_total -= 1

    def _sampler(self):
        while True:
            yield self.sim.timeout(1.0)
            for vm in self.vms:
                a = vm.acc
                a.s_inflight += vm.threads.count + len(vm.threads.queue)
                a.s_wait += len(vm.threads.queue)
                a.s_runq += len(vm.cpu.queue)
                a.n += 1

    # ---------- control step ----------
    def step(self, delta_total: int) -> Observation:
        if delta_total:
            self._apply(int(delta_total))
        self.t += self.dt
        self.sim.run(until=self.t)
        return self._observe(int(delta_total))

    def _observe(self, delta: int) -> Observation:
        dt, app = self.dt, self.app
        acc = {k: [] for k in ("busy", "io", "rate", "cmpl", "inflight", "runq", "wait", "netin", "netout", "iops")}
        for vm in self.vms:
            a = vm.acc; n = max(a.n, 1)
            acc["busy"].append(min(a.cpu_time / (app.vcpu * dt), 1.0))
            acc["io"].append(min(a.io_time / (max(app.io_cap, 1) * dt), 1.0) if vm.io is not None else 0.0)
            acc["rate"].append(a.arrivals / dt); acc["cmpl"].append(a.completions / dt)
            acc["inflight"].append(a.s_inflight / n); acc["wait"].append(a.s_wait / n); acc["runq"].append(a.s_runq / n)
            acc["netin"].append(a.kb_in / dt); acc["netout"].append(a.kb_out / dt); acc["iops"].append(a.iops / dt)
            a.reset()
        d = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}
        d["pkts"] = (d["netin"] + d["netout"]) / 1.4
        d["mem"] = app.mem_mb_per_req * d["inflight"]
        metrics = self.gen.generate(d, self.w)
        rts = self._rt_int
        rt_mean = float(np.mean(rts)) if rts else float("nan")
        n_done = len(rts)
        self._rt_int = []
        rr = self._arr_int / dt
        self._arr_int = 0
        cx = self.wl.complexity_at(self.t)
        obs = Observation(t=self.t, w=self.w, pending=self.pending, metrics=metrics, metric_names=self.gen.names,
                          rt_mean_ms=rt_mean, n_completed=n_done, cpu_pct=100.0 * d["busy"], rr=rr,
                          hidden_complexity=cx)
        self._rows.append(dict(t=self.t, w=self.w, pending=self.pending, delta=delta, rr=rr, rt_mean_ms=rt_mean,
                               n_completed=n_done, cpu_pct=obs.cpu_pct, complexity=cx,
                               inflight=d["inflight"] * self.w))
        return obs

    def finalize(self):
        """Stop arrivals, drain the backlog, return (interval_df, request_df)."""
        self._arrivals_on = False
        tries = 0
        while self._inflight_total > 0 and tries < 120:
            self.sim.run(until=self.sim.now + 60.0); tries += 1
        req = pd.DataFrame({"t0": self._req_t0, "rt_ms": self._req_rt, "w_at_arrival": self._req_w})
        return pd.DataFrame(self._rows), req


def make_env(cfg: dict, app: str, workload: str, seed: int, duration_s: float | None = None, **app_over) -> SimpyEnv:
    app_cfg = dict(cfg["apps"][app]); app_cfg.update(app_over)
    dur = duration_s if duration_s is not None else cfg["duration_s"]
    wl = build_workload(cfg["workloads"][workload], app_cfg, dur, seed)
    return SimpyEnv(app_cfg, wl, cfg, seed, dur)
