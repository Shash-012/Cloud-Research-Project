from pathlib import Path

import pytest

from autoscale.config import app_capacity_per_vm, load_config


ROOT = Path(__file__).parents[1]


def test_load_config_includes_default_sections() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")

    assert config["interval_s"] == 10
    assert config["max_vms"] == 14
    assert config["alpha"] == 0.001
    assert config["apps"]
    assert config["workloads"]


def test_load_config_keeps_app_capacity_calculation() -> None:
    app = {
        "cpu_ms": 20,
        "io_ms": 5,
        "vcpu": 1,
        "io_cap": 100,
        "slots": 100,
    }

    assert app_capacity_per_vm(app) == pytest.approx(50.0)


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
