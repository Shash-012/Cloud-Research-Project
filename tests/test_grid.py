import csv

import pandas as pd

from autoscaler.config import default_experiments_config
from autoscaler.eval.grid import Job, Manifest, expand, is_complete


def test_grid_sizes_match_roadmap():
    exp = default_experiments_config()
    assert len(expand(exp, "cloudsim", "main")) == (3 + 7) * 4 * 10          # 400
    assert len(expand(exp, "cloudsim", "ablations")) == 10 * 2 * 10          # 200
    assert len(expand(exp, "cloudsim", "sensitivity")) == 2 * 6 * 10         # 120


def test_run_ids_unique_and_readable():
    exp = default_experiments_config()
    jobs = [j for part in ("main", "ablations", "sensitivity") for j in expand(exp, "cloudsim", part)]
    ids = [j.run_id for j in jobs]
    assert len(ids) == len(set(ids))
    assert "cloudsim__proposed_w111__flash_crowd__s3" in ids
    assert "cloudsim__opt__daily_ramp_burst__s1__pw_square_r1.42" in ids


def test_smoke_filters_and_power():
    exp = default_experiments_config()
    jobs = expand(exp, "synthetic", "main", ["opt", "proposed_w111"], ["step"], [1, 2])
    assert [j.run_id for j in jobs] == ["synthetic__opt__step__s1", "synthetic__opt__step__s2",
                                        "synthetic__proposed_w111__step__s1", "synthetic__proposed_w111__step__s2"]
    p = Job("cloudsim", "sensitivity", "opt", "step", 1, "square", 1.42).power(400.0)
    assert abs((p["p_max_w"] - p["p_idle_w"]) / p["p_idle_w"] - 1.42) < 1e-12


def test_completion_check_and_manifest(tmp_path):
    folder = tmp_path / "r"
    folder.mkdir()
    assert not is_complete(folder, 3)
    pd.DataFrame({"step": [0, 1, 2]}).to_parquet(folder / "run_log.parquet")
    (folder / "done.json").write_text('{"status": "ok"}')
    assert is_complete(folder, 3) and not is_complete(folder, 4)
    m = Manifest(tmp_path / "manifest.csv")
    m.append(Job("synthetic", "main", "opt", "step", 1), 3, 1.5, "ok", folder)
    rows = list(csv.DictReader(open(tmp_path / "manifest.csv")))
    assert rows[0]["run_id"] == "synthetic__opt__step__s1" and rows[0]["status"] == "ok"
