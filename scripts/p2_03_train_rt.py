"""Stage 2: hyperparameter search, deep ensemble, MSE baseline net (Step 2.6) and quantile models (2.7)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from autoscaler.config import load_config
from autoscaler.data.dataset import load_transitions
from autoscaler.data.splits import split_from_config
from autoscaler.logging_utils import run_dir
from autoscaler.models.metric_filters import load_selected
from autoscaler.models.quantile import QuantileHGB, QuantileNetModel
from autoscaler.models.rt_ensemble import RTEnsemble, stage2_columns
from autoscaler.models.tuning import load_hparams, random_search, save_hparams


def accuracy(ens: RTEnsemble, test: pd.DataFrame, slo_cap_ms: float = 500.0) -> dict:
    """Ensemble-mean accuracy with TRUE post-scaling metrics as inputs."""
    mu, _ = ens.predict_frame(test)
    m = mu.mean(axis=0)
    y = np.log(test["rt_after_ms"].to_numpy())
    rt = test["rt_after_ms"].to_numpy()
    low = rt < slo_cap_ms
    return {"r2_log": float(r2_score(y, m)),
            "mae_ms_below_500": float(np.mean(np.abs(np.exp(m[low]) - rt[low]))),
            "rows_below_500": int(low.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--data", default="data/processed/synthetic_transitions.parquet")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--skip-search", action="store_true", help="reuse models/rt_hparams.json")
    ap.add_argument("--include-rate", action="store_true", help="also train the abl_with_rate ensemble")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rd = run_dir(args.out, "p2_03_train_rt", cfg)
    model_dir, tables = Path(args.model_dir), Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)

    train, _, test = split_from_config(load_transitions(args.data), cfg)
    metrics = load_selected(model_dir / "selected_metrics.json")
    cols = stage2_columns(metrics, cfg["ensemble"]["include_request_rate"])
    print("stage-2 inputs:", cols)

    if args.skip_search:
        hp = load_hparams(model_dir / "rt_hparams.json")
    else:
        hp, trials = random_search(train, cols, cfg, n_jobs=args.jobs)
        trials.to_csv(rd / "search_trials.csv", index=False)
        save_hparams(hp, model_dir / "rt_hparams.json")
        print(trials.sort_values("val_nll").head(5).to_string(index=False))
    print("hparams:", hp)

    ens = RTEnsemble(cols, hp).fit(train, cfg, n_jobs=args.jobs)
    ens.save(model_dir / "rt_ensemble")
    mse = RTEnsemble(cols, hp, kind="mean").fit(train, cfg, members=1, seed=cfg["ensemble"]["seed"])
    mse.save(model_dir / "rt_mse")

    rows = [{"model": "ensemble", "members": ens.members, **accuracy(ens, test)},
            {"model": "ensemble_member1", "members": 1, **accuracy(ens.subset(1), test)},
            {"model": "mse_net", "members": 1, **accuracy(mse, test)}]

    if args.include_rate:
        cols_r = stage2_columns(metrics, include_rate=True)
        ens_r = RTEnsemble(cols_r, hp).fit(train, cfg, n_jobs=args.jobs)
        ens_r.save(model_dir / "with_rate" / "rt_ensemble")
        rows.append({"model": "ensemble_with_rate", "members": ens_r.members, **accuracy(ens_r, test)})

    table = pd.DataFrame(rows)
    table.to_csv(tables / "stage2_accuracy.csv", index=False)
    table.to_csv(rd / "stage2_accuracy.csv", index=False)
    (rd / "train_info.json").write_text(json.dumps(ens.train_info, indent=2))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # Step 2.7: quantile models for the ablation (tau = 1 - delta), true after-metrics on test
    q = cfg["quantile"]
    qnet = QuantileNetModel(cols, hp, q["tau"]).fit(train, cfg, seed=q["seed"])
    qnet.save(model_dir / "quantile_nn")
    qhgb = QuantileHGB(cols, q["tau"], q["hgb"]["max_iter"], q["hgb"]["learning_rate"], q["seed"]).fit(train)
    qhgb.save(model_dir / "quantile_hgb.joblib")
    y = np.log(test["rt_after_ms"].to_numpy())
    qrows = [{"model": name, "tau": q["tau"], "share_y_le_q": float(np.mean(y <= mdl.predict_frame(test)))}
             for name, mdl in (("quantile_nn", qnet), ("quantile_hgb", qhgb))]
    qtable = pd.DataFrame(qrows)
    qtable.to_csv(tables / "quantile_coverage.csv", index=False)
    qtable.to_csv(rd / "quantile_coverage.csv", index=False)
    print(qtable.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
