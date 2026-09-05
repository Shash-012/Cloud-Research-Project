"""Response-time predictor with an ensemble uncertainty estimate.

The base paper uses a single two-layer feedforward network (LeakyReLU +
dropout) and takes its output as a point estimate. This is the first of the two
extension points: we train N independently initialised networks on bootstrap
resamples of the calibration data and read

    mean  of member predictions -> the point estimate
    stdev of member predictions -> the uncertainty

Deviation from the paper, stated plainly: sklearn's MLPRegressor offers neither
LeakyReLU nor dropout. We use ReLU with L2 (`alpha`) regularisation instead.
Nothing in the extension depends on the activation choice; if the eventual
CloudSim build wants the paper's exact architecture it moves to PyTorch.

Targets are trained in log space. Response time under a queueing law is
positive and heavy-tailed -- a saturated pool produces multi-second values that
are three orders of magnitude above the SLO. Fitting those directly under a
squared-error loss would spend the whole model on the overloaded regime and
leave it imprecise exactly where the decision boundary is. Each member's
prediction is exponentiated back to milliseconds *before* the ensemble is
aggregated, so the reported mean and standard deviation are both in ms and the
decision rule needs no change.

Two kinds of uncertainty, and the decision rule needs both:

  epistemic -- disagreement between ensemble members. Large where training data
    was sparse or inconsistent; shrinks with more data. This is what the
    ensemble gives directly.

  aleatoric -- irreducible noise in the environment itself. Here it comes from
    the unobserved request-complexity drift and from observation noise. Every
    member sees the same noise and learns the same conditional mean, so the
    ensemble spread does *not* contain it.

Using ensemble spread alone as sigma is a trap, and an easy one to fall into:
in the well-sampled region the members agree to within a millisecond or two, so
`mean + 1.65*sigma` is barely above `mean`. A controller built on that
constraint parks itself exactly on the SLO boundary and then violates the SLO
roughly half the time, while reporting that it is running at the 95th
percentile. It is confidently wrong precisely where it matters.

So the aleatoric part is estimated too. Training happens in log space, where the
environment's noise is close to multiplicative, so residual scale is estimated
once on a held-out slice as a single log-space standard deviation and turned
back into milliseconds as sigma_a = mean * sigma_log. The two combine in
quadrature:

    sigma_total = sqrt(sigma_epistemic**2 + (mean * sigma_log)**2)

This is a coarse calibration -- one global noise scale, not a per-region one.
It is enough to make the chance constraint mean roughly what it says. A proper
calibration study (conditional variance, coverage checks against nominal
levels) is a later milestone.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import config as C


@dataclass
class Prediction:
    mean: np.ndarray
    std: np.ndarray  # total predictive sigma: epistemic and aleatoric combined


class EnsembleRTPredictor:
    def __init__(self, n_models: int = C.N_ENSEMBLE, base_seed: int = C.ENSEMBLE_BASE_SEED,
                 log_target: bool = True, noise_holdout: float = 0.2):
        self.n_models = n_models
        self.base_seed = base_seed
        self.log_target = log_target
        self.noise_holdout = noise_holdout
        self.models: list[MLPRegressor] = []
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.sigma_log = 0.0  # aleatoric scale, set during fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleRTPredictor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        # Hold out a slice for the noise estimate. Members never see it, so the
        # residuals on it reflect real predictive error rather than memorised
        # rows -- an in-sample estimate would badly understate the noise.
        rng = np.random.default_rng(self.base_seed)
        perm = rng.permutation(len(X))
        n_hold = max(1, int(self.noise_holdout * len(X)))
        hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]
        X_fit, y_fit = X[fit_idx], y[fit_idx]

        Xs = self.x_scaler.fit_transform(X_fit)
        ys = self.y_scaler.fit_transform(np.log(y_fit) if self.log_target else y_fit).ravel()

        self.models = []
        for i in range(self.n_models):
            # Two independent sources of member diversity: a bootstrap resample
            # of the rows, and a different weight initialisation.
            idx = rng.integers(0, len(Xs), size=len(Xs))
            model = MLPRegressor(
                hidden_layer_sizes=C.HIDDEN_LAYERS,
                activation="relu",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=C.MAX_ITER,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=self.base_seed + i,
            )
            model.fit(Xs[idx], ys[idx])
            self.models.append(model)

        # Aleatoric scale: spread of held-out residuals in log space, with the
        # epistemic part removed so the two are not double counted.
        held = self.predict(X[hold_idx], _epistemic_only=True)
        resid_log = np.log(y[hold_idx].ravel()) - np.log(np.maximum(held.mean, 1e-6))
        epi_log = held.std / np.maximum(held.mean, 1e-6)
        total_var = float(np.var(resid_log))
        self.sigma_log = float(np.sqrt(max(total_var - float(np.mean(epi_log ** 2)), 1e-6)))
        return self

    def predict(self, X: np.ndarray, _epistemic_only: bool = False) -> Prediction:
        """Return per-row predictive mean and total standard deviation, in ms."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xs = self.x_scaler.transform(X)

        # Invert the target transform per member, then aggregate, so that the
        # spread is reported in milliseconds rather than in scaled or log units.
        members = []
        for m in self.models:
            z = self.y_scaler.inverse_transform(m.predict(Xs).reshape(-1, 1)).ravel()
            members.append(np.exp(z) if self.log_target else z)
        members = np.stack(members)
        mean = members.mean(axis=0)
        epistemic = members.std(axis=0)
        if _epistemic_only or self.sigma_log <= 0.0:
            return Prediction(mean=mean, std=epistemic)
        aleatoric = mean * self.sigma_log
        return Prediction(mean=mean, std=np.sqrt(epistemic ** 2 + aleatoric ** 2))

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "EnsembleRTPredictor":
        return joblib.load(path)
