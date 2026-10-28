"""Bridge messages: newline-delimited JSON over one TCP connection (Step 3.11).

Java -> Python  {"type":"hello","run_id":...,"policy":...,"seed":...,"config":{...}}
Python -> Java  {"type":"ready"}
Java -> Python  {"type":"observe","step":...,"time_s":...,"n_running":...,"n_booting":...,
                 "metrics":{...},"rt_mean_ms":...}            (also "n_draining" and "rate")
Python -> Java  {"type":"action","k":...}                       (also "fallback")
Java -> Python  {"type":"bye","summary":{...}}
Python -> Java  {"type":"error","message":...}                  (on a Python-side failure)
Missing measurements (an interval without completions) are sent as null.
"""
import json
import math

REQUIRED = {
    "hello": {"run_id": str, "policy": str, "seed": int, "config": dict},
    "observe": {"step": int, "time_s": (int, float), "n_running": int, "n_booting": int, "metrics": dict},
    "bye": {"summary": dict},
    "ready": {},
    "action": {"k": int},
    "error": {"message": str},
}


class ProtocolError(ValueError):
    pass


def validate(msg: dict) -> dict:
    if not isinstance(msg, dict) or "type" not in msg:
        raise ProtocolError(f"message without type: {msg!r}")
    kind = msg["type"]
    if kind not in REQUIRED:
        raise ProtocolError(f"unknown message type {kind!r}")
    for field, typ in REQUIRED[kind].items():
        if field not in msg:
            raise ProtocolError(f"{kind} message lacks {field!r}")
        if isinstance(msg[field], bool) or not isinstance(msg[field], typ):
            raise ProtocolError(f"{kind}.{field} has type {type(msg[field]).__name__}")
    return msg


def encode(msg: dict) -> bytes:
    return (json.dumps(validate(msg), allow_nan=False) + "\n").encode("utf-8")


def decode(line: bytes | str) -> dict:
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    try:
        return validate(json.loads(line))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc


def ready() -> dict:
    return {"type": "ready"}


def action(k: int, fallback: bool = False) -> dict:
    return {"type": "action", "k": int(k), "fallback": bool(fallback)}


def error(message: str) -> dict:
    return {"type": "error", "message": str(message)}


def _num(v):
    return math.nan if v is None else float(v)


def observation(msg: dict) -> dict:
    """Turn an observe message into the obs dict the Phase 2 policies expect."""
    obs = {m: _num(v) for m, v in msg["metrics"].items()}
    obs.update(step=msg["step"], time_s=msg["time_s"], n_running=msg["n_running"], n_booting=msg["n_booting"],
               n_draining=msg.get("n_draining", 0), rt_mean_ms=_num(msg.get("rt_mean_ms")),
               rate=_num(msg.get("rate")))
    return obs
