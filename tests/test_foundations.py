import numpy as np

from autoscaler import schema
from autoscaler.config import CONFIG_DIR, get, load_config
from autoscaler.logging_utils import run_dir
from autoscaler.seeds import child_rngs, seed_everything


def test_load_config_reads_both_files():
    proto = load_config(CONFIG_DIR / "prototype.yaml")
    exp = load_config(CONFIG_DIR / "experiments.yaml")
    assert get(proto, "slo.rt_ms") == 250.0
    assert get(proto, "system.vm_max") == 30
    assert get(proto, "missing.key", "x") == "x"
    assert exp["eval_seeds"] == list(range(1, 11))


def test_seed_everything_reproducible():
    a = seed_everything(5).normal(size=3)
    b = seed_everything(5).normal(size=3)
    assert np.array_equal(a, b)


def test_calibration_and_eval_seeds_disjoint():
    proto = load_config(CONFIG_DIR / "prototype.yaml")
    exp = load_config(CONFIG_DIR / "experiments.yaml")
    eval_seeds = set(exp["eval_seeds"])
    assert not eval_seeds & set(exp["calibration_seeds"])
    assert proto["calibration"]["seed"] not in eval_seeds


def test_child_streams_independent_and_reproducible():
    a1, b1 = (g.normal(size=4) for g in child_rngs(3, 2))
    a2, b2 = (g.normal(size=4) for g in child_rngs(3, 2))
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert not np.array_equal(a1, b1)


def test_schema_lists_have_no_duplicates():
    for cols in (schema.TRANSITION_COLUMNS, schema.RUN_LOG_COLUMNS,
                 schema.DECISION_CANDIDATE_FIELDS, schema.METRICS):
        assert len(cols) == len(set(cols))


def test_run_dir_writes_config_and_rev(tmp_path):
    p = run_dir(tmp_path, "demo", {"a": 1})
    assert (p / "config.yaml").exists() and (p / "git_rev.txt").exists()
    assert load_config(p / "config.yaml") == {"a": 1}
