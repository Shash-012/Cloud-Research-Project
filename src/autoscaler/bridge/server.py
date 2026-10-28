"""Bridge server: answers the simulator's observations with the Phase 2 policies (Step 3.11)."""
import socket
import threading
import traceback
from pathlib import Path


from ..decision.policies import load_models, make_policy
from ..seeds import STREAM_CONTROLLER, child_rngs
from ..sim.loop import save_decisions
from . import protocol


class BridgeServer:
    """Accepts one connection, serves one run, writes its decision log, then exits.

    `cfg` is the merged config (config.merge_cloudsim) and `models` the loaded CloudSim models (or None for
    policies that need none). If Java sends nothing for `timeout_s`, the server gives up instead of hanging.
    """

    def __init__(self, cfg: dict, exp_cfg: dict, models: dict | None, out_dir, host: str = "127.0.0.1",
                 port: int = 0, timeout_s: float = 120.0):
        self.cfg, self.exp_cfg, self.models = cfg, exp_cfg, models
        self.out_dir = Path(out_dir)
        self.timeout_s = timeout_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.sock.settimeout(timeout_s)
        self.port = self.sock.getsockname()[1]
        self.decisions: list[dict] = []
        self.summary: dict | None = None
        self.error: str | None = None
        self.hello: dict | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> "BridgeServer":
        self._thread = threading.Thread(target=self.serve, name=f"bridge-{self.port}", daemon=True)
        self._thread.start()
        return self

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def serve(self) -> None:
        conn = None
        try:
            conn, _ = self.sock.accept()
            conn.settimeout(self.timeout_s)
            reader = conn.makefile("rb")
            policy = None
            for line in reader:
                msg = protocol.decode(line)
                kind = msg["type"]
                if kind == "hello":
                    self.hello = msg
                    policy = make_policy(msg["policy"], self.cfg, self.exp_cfg, self.models)
                    policy.reset(child_rngs(int(msg["seed"]), 3)[STREAM_CONTROLLER])
                    state: dict = {}
                    conn.sendall(protocol.encode(protocol.ready()))
                elif kind == "observe":
                    if policy is None:
                        raise protocol.ProtocolError("observe before hello")
                    obs = protocol.observation(msg)
                    k, info = policy.decide(obs, state)
                    fallback = bool(info.get("fallback", False))
                    self.decisions.append({"step": obs["step"], "w": obs["n_running"] + obs["n_booting"],
                                           "k": int(k), "fallback": fallback, "chosen_U": info.get("chosen_U"),
                                           "chosen_mean_log": info.get("chosen_mean_log"),
                                           "candidates": info.get("candidates", [])})
                    conn.sendall(protocol.encode(protocol.action(k, fallback)))
                elif kind == "bye":
                    self.summary = msg["summary"]
                    break
                else:
                    raise protocol.ProtocolError(f"unexpected message {kind}")
        except Exception as exc:              # log the traceback to the run folder; tell Java if possible
            self.error = f"{type(exc).__name__}: {exc}"
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "bridge_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if conn is not None:
                try:
                    conn.sendall(protocol.encode(protocol.error(self.error)))
                except OSError:
                    pass
        finally:
            if self.decisions:
                save_decisions(self.decisions, self.out_dir)
            if conn is not None:
                conn.close()
            self.sock.close()


def serve_run(cfg, exp_cfg, model_dir, out_dir, policy: str, timeout_s: float = 120.0, port: int = 0):
    """Start a bridge for one run in a background thread; models are loaded only if the policy needs them."""
    needs_models = policy not in ("cpu_threshold", "rt_threshold")
    models = load_models(model_dir) if needs_models else None
    return BridgeServer(cfg, exp_cfg, models, out_dir, port=port, timeout_s=timeout_s).start()

