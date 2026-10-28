import gzip
import json
import shutil
import socket
from pathlib import Path

import pytest

from autoscaler.bridge import protocol
from autoscaler.bridge.server import BridgeServer
from autoscaler.config import (default_cloudsim_config, default_experiments_config, default_prototype_config,
                               java_config, merge_cloudsim)


@pytest.fixture
def cfgs():
    cs, proto, exp = default_cloudsim_config(), default_prototype_config(), default_experiments_config()
    return cs, proto, exp, merge_cloudsim(proto, cs, exp)


def test_protocol_roundtrip_and_validation():
    msg = {"type": "observe", "step": 3, "time_s": 240.0, "n_running": 5, "n_booting": 1,
           "metrics": {"cpu_util": 70.0, "queue_len": None}, "rt_mean_ms": None, "rate": 120.0}
    back = protocol.decode(json.dumps(msg))
    obs = protocol.observation(back)
    assert obs["n_running"] == 5 and obs["cpu_util"] == 70.0
    assert obs["rt_mean_ms"] != obs["rt_mean_ms"] and obs["queue_len"] != obs["queue_len"]   # NaN
    assert protocol.decode(protocol.encode(protocol.action(2, True))) == {"type": "action", "k": 2, "fallback": True}
    for bad in ('{"step": 1}', '{"type": "nope"}', '{"type": "action", "k": "2"}', '{"type": "action", "k": true}',
                'not json'):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(bad)


def test_merge_cloudsim_scales_slo(cfgs):
    cs, proto, exp, cfg = cfgs
    cs2 = json.loads(json.dumps(cs))
    cs2["requests"]["capacity_scale"] = 0.25
    m = merge_cloudsim(proto, cs2, exp)
    assert m["slo"]["rt_ms"] == pytest.approx(1000.0)
    assert m["system"]["vm_max"] == 30 and m["actions"] == cs["actions"]
    assert m["selection"] == proto["selection"]
    assert java_config(cs, exp, 6001)["bridge"]["port"] == 6001


def _client(port, messages):
    replies = []
    with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
        f = s.makefile("rb")
        for m in messages:
            s.sendall((json.dumps(m) + "\n").encode())
            if m["type"] != "bye":
                replies.append(json.loads(f.readline()))
    return replies


def test_server_serves_a_run_and_writes_decisions(cfgs, tmp_path):
    cs, proto, exp, cfg = cfgs
    server = BridgeServer(cfg, exp, None, tmp_path, timeout_s=10).start()
    obs = lambda step, rt: {"type": "observe", "step": step, "time_s": 60.0 * (step + 1), "n_running": 4,
                            "n_booting": 0, "metrics": {"cpu_util": 50.0}, "rt_mean_ms": rt, "rate": 90.0}
    msgs = [{"type": "hello", "run_id": "r", "policy": "rt_threshold", "seed": 3, "config": {}},
            obs(0, 240.0), obs(1, 240.0), obs(2, None), {"type": "bye", "summary": {"steps": 3}}]
    replies = _client(server.port, msgs)
    server.join(10)
    assert replies[0] == {"type": "ready"}
    assert [r["k"] for r in replies[1:]] == [1, 0, 0]        # +1 above 0.8 SLO, then cooldown
    assert server.summary == {"steps": 3} and server.error is None
    with gzip.open(tmp_path / "decisions.jsonl.gz", "rt") as f:
        assert len(f.readlines()) == 3


def test_server_reports_errors_to_java_and_logs_traceback(cfgs, tmp_path):
    cs, proto, exp, cfg = cfgs
    server = BridgeServer(cfg, exp, None, tmp_path, timeout_s=10).start()
    replies = _client(server.port, [{"type": "hello", "run_id": "r", "policy": "no_such_policy", "seed": 1,
                                     "config": {}}])
    server.join(10)
    assert replies[0]["type"] == "error" and "no_such_policy" in replies[0]["message"]
    assert (tmp_path / "bridge_error.txt").exists()


def test_server_times_out_without_a_client(cfgs, tmp_path):
    cs, proto, exp, cfg = cfgs
    server = BridgeServer(cfg, exp, None, tmp_path, timeout_s=0.5).start()
    server.join(5)
    assert server.error is not None and "timed out" in server.error.lower()


needs_java = pytest.mark.skipif(shutil.which("java") is None, reason="java not installed")


@pytest.mark.cloudsim
@needs_java
def test_java_and_python_thresholds_make_identical_runs(cfgs, tmp_path):
    """Step 3.10: the Java baselines decide like their Python twins (same seed -> byte-identical run logs)."""
    from autoscaler.sim.cloudsim import run_cloudsim
    cs, proto, exp, cfg = cfgs
    for pol in ("cpu_threshold", "rt_threshold"):
        logs = []
        for remote in (False, True):
            r = run_cloudsim(pol, "sawtooth", 5, tmp_path / f"{pol}_{remote}", proto, cs, exp, steps=12,
                             remote=remote)
            assert r["status"] == "ok", r
            logs.append((Path(r["run_dir"]) / "run_log.csv").read_bytes())
        assert logs[0] == logs[1]


@pytest.mark.cloudsim
@needs_java
def test_java_aborts_when_python_never_answers(cfgs, tmp_path):
    """Step 3.11: a silent Python side makes Java fail with a clear error instead of hanging."""
    import subprocess
    from autoscaler.sim.cloudsim import ensure_jar, workload_csv
    cs, proto, exp, cfg = cfgs
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)                       # accepts the connection but never replies
    port = listener.getsockname()[1]
    jcfg = java_config(cs, exp, port)
    jcfg["bridge"]["timeout_s"] = 2
    (tmp_path / "c.json").write_text(json.dumps(jcfg))
    wl = workload_csv("step", 1, cs["time"]["steps_per_run"], proto["workloads"])
    proc = subprocess.run(["java", "-jar", str(ensure_jar()), "--config", str(tmp_path / "c.json"),
                           "--workload-csv", str(wl), "--policy", "proposed", "--out", str(tmp_path),
                           "--port", str(port), "--steps", "3"], capture_output=True, text=True, timeout=60)
    listener.close()
    assert proc.returncode == 3
    assert "bridge" in proc.stderr and "timeout" in proc.stderr.lower()
