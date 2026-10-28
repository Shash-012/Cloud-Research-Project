"""Random hyperparameter search for one ensemble member (Step 2.6)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .rt_ensemble import RTEnsemble


def sample_hparams(space: dict, rng: np.random.Generator) -> dict:
    def loguniform(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return {
        "hidden": list(space["hidden"][rng.integers(len(space["hidden"]))]),
        "lr": loguniform(*space["lr"]),
        "dropout": float(space["dropout"][rng.integers(len(space["dropout"]))]),
        "weight_decay": loguniform(*space["weight_decay"]),
        "batch_size": int(space["batch_size"][rng.integers(len(space["batch_size"]))]),
        "slope": float(space["slope"][rng.integers(len(space["slope"]))]),
    }


def _trial(df, columns, cfg, hp, seed):
    import torch
    torch.set_num_threads(1)
    return RTEnsemble(columns, hp).fit(df, cfg, members=1, seed=seed).train_info[0]


def random_search(df: pd.DataFrame, columns, cfg: dict, seed: int | None = None,
                  trials: int | None = None, n_jobs: int = 1):
    """Train one member per trial; score by validation NLL. Returns (best_hparams, trial table)."""
    from joblib import Parallel, delayed
    e = cfg["ensemble"]
    seed = e["tuning_seed"] if seed is None else seed
    trials = e["search_trials"] if trials is None else trials
    rng = np.random.default_rng(seed)
    hps = [sample_hparams(cfg["tuning"], rng) for _ in range(trials)]
    infos = Parallel(n_jobs=n_jobs)(delayed(_trial)(df, columns, cfg, hp, seed) for hp in hps)
    rows = []
    for i, (hp, info) in enumerate(zip(hps, infos)):
        rows.append({"trial": i, **hp, "hidden": tuple(hp["hidden"]),
                     "val_nll": info["val_loss"], "epochs": info["epochs"]})
    table = pd.DataFrame(rows)
    best = table.loc[table["val_nll"].idxmin()]
    best_hp = {"hidden": list(best["hidden"]), "lr": float(best["lr"]), "dropout": float(best["dropout"]),
               "weight_decay": float(best["weight_decay"]), "batch_size": int(best["batch_size"]),
               "slope": float(best["slope"])}
    return best_hp, table


def save_hparams(hp: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(hp, indent=2))


def load_hparams(path) -> dict:
    return json.loads(Path(path).read_text())
