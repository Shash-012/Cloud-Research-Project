"""Stage 1: per-metric post-scaling estimator chosen by cross-validation (Step 2.5, E1)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..schema import clip_metric
from .features import FEATURES, transition_inputs

# candidate -> feature set; the order is simplest first (ties go to the earlier one)
CANDIDATE_FEATURES = {"paper_linear": "paper", "basis_linear": "basis",
                      "hgb": "basis", "mlp_ensemble": "basis"}


class MLPEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Average of `members` small MLPs on standardised inputs and target."""

    def __init__(self, members=5, hidden=(32, 16), max_iter=300, early_stopping=True, seed=0):
        self.members = members
        self.hidden = hidden
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.seed = seed

    def fit(self, X, y):
        self.models_ = []
        for j in range(self.members):
            mlp = MLPRegressor(hidden_layer_sizes=tuple(self.hidden), max_iter=self.max_iter,
                               early_stopping=self.early_stopping, random_state=self.seed + j)
            model = TransformedTargetRegressor(make_pipeline(StandardScaler(), mlp),
                                               transformer=StandardScaler())
            self.models_.append(model.fit(X, y))
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)


def make_candidate(name: str, cfg: dict):
    s = cfg["stage1"]
    if name in ("paper_linear", "basis_linear"):
        return LinearRegression()
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=s["hgb"]["max_iter"],
                                             learning_rate=s["hgb"]["learning_rate"],
                                             random_state=s["cv_seed"])
    if name == "mlp_ensemble":
        m = s["mlp"]
        return MLPEnsembleRegressor(members=m["members"], hidden=tuple(m["hidden"]),
                                    max_iter=m["max_iter"], early_stopping=m["early_stopping"],
                                    seed=m["seed"])
    raise ValueError(f"unknown stage-1 candidate {name}")


def _fit_metric(df: pd.DataFrame, metric: str, candidates, cfg: dict):
    """CV every candidate for one metric; return scores, chosen name, refit model, OOF preds."""
    s = cfg["stage1"]
    m, w, k = transition_inputs(df, metric)
    y = df[f"{metric}_after"].to_numpy(float)
    folds = list(KFold(s["cv_folds"], shuffle=True, random_state=s["cv_seed"]).split(y))
    scores, oofs = {}, {}
    for name in candidates:
        X = FEATURES[CANDIDATE_FEATURES[name]](m, w, k)
        oof = np.empty_like(y)
        fold_r2 = []
        for tr, te in folds:
            est = clone(make_candidate(name, cfg)).fit(X[tr], y[tr])
            oof[te] = clip_metric(metric, est.predict(X[te]))
            fold_r2.append(r2_score(y[te], oof[te]))
        scores[name] = float(np.mean(fold_r2))
        oofs[name] = oof
    best = max(scores.values())
    chosen = next(c for c in candidates if scores[c] >= best - s["tie_tol"])
    X = FEATURES[CANDIDATE_FEATURES[chosen]](m, w, k)
    model = make_candidate(chosen, cfg).fit(X, y)
    return metric, scores, chosen, model, oofs[chosen]


class MetricEstimator:
    """One estimator per metric. `candidates` restricts the choice (e.g. ["paper_linear"])."""

    def __init__(self, cfg: dict, candidates=None):
        self.cfg = cfg
        self.candidates = list(candidates or cfg["stage1"]["candidates"])
        self.models: dict = {}       # metric -> fitted model
        self.chosen: dict = {}       # metric -> candidate name
        self.scores: dict = {}       # metric -> {candidate: cv r2}
        self.oof_residuals: dict = {}  # metric -> out-of-fold residual vector (training rows)
        self.metrics: list[str] = []   # selected metrics (prediction output order)

    def fit(self, df: pd.DataFrame, metrics, n_jobs: int = 1):
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_metric)(df, m, self.candidates, self.cfg) for m in metrics)
        for metric, scores, chosen, model, oof in results:
            self.scores[metric] = scores
            self.chosen[metric] = chosen
            self.models[metric] = model
            self.oof_residuals[metric] = df[f"{metric}_after"].to_numpy(float) - oof
        self.metrics = list(metrics)
        return self

    def restrict(self, metrics):
        """Set the selected metrics: prediction columns and residual matrix follow this order."""
        missing = set(metrics) - set(self.models)
        if missing:
            raise ValueError(f"metrics not fitted: {sorted(missing)}")
        self.metrics = list(metrics)
        return self

    @property
    def residuals(self) -> np.ndarray:
        """Out-of-fold residual matrix, rows by selected metrics."""
        return np.column_stack([self.oof_residuals[m] for m in self.metrics])

    @property
    def residual_cov(self) -> np.ndarray:
        return np.atleast_2d(np.cov(self.residuals, rowvar=False))

    def _predict_metric(self, metric, m, w, k):
        X = FEATURES[CANDIDATE_FEATURES[self.chosen[metric]]](m, w, k)
        return clip_metric(metric, self.models[metric].predict(X))

    def predict(self, before: dict, w: int, ks, metrics=None) -> np.ndarray:
        """Predicted post-scaling metrics, shape (len(ks), n_metrics)."""
        ks = np.asarray(ks, float)
        ws = np.full_like(ks, float(w))
        cols = metrics or self.metrics
        return np.column_stack([self._predict_metric(c, np.full_like(ks, float(before[c])), ws, ks)
                                for c in cols])

    def predict_frame(self, df: pd.DataFrame, metrics=None) -> np.ndarray:
        """Row-wise predictions for a transition table, shape (rows, n_metrics)."""
        cols = metrics or self.metrics
        return np.column_stack([self._predict_metric(c, *transition_inputs(df, c)) for c in cols])

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "MetricEstimator":
        return joblib.load(path)
