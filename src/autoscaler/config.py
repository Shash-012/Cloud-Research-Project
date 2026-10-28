"""YAML configuration loading and dotted access."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(cfg: dict, key: str, default=None):
    """Dotted access: get(cfg, "a.b.c")."""
    node = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def save_config(cfg: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def default_prototype_config() -> dict:
    return load_config(CONFIG_DIR / "prototype.yaml")


def default_experiments_config() -> dict:
    return load_config(CONFIG_DIR / "experiments.yaml")
