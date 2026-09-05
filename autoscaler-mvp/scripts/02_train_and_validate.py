"""Step 2: fit the metric estimator and the RT ensemble, then validate them.

Writes models/*.joblib and two validation figures.

Success criteria (checked and printed, per karpathy-guidelines -- do not move
on to the control loop until both hold):
  * metric estimator: at least one metric retained by the R^2 > 0.9 filter, and
    every retained metric confirmed above the threshold on held-out rows
  * RT ensemble:      R^2 > 0.7 on held-out calibration rows

The predicted-vs-actual scatter mirrors Fig. 5/6 of the base paper, so it is
saved as an artifact rather than shown and discarded.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from _bootstrap import DATA, FIGURES, MODELS  # noqa: E402

from autoscaler import config as C  # noqa: E402
from autoscaler.calibration import rt_training_frame  # noqa: E402
from autoscaler.metric_estimator import MetricEstimator  # noqa: E402
from autoscaler.predictor import EnsembleRTPredictor  # noqa: E402

import joblib  # noqa: E402

MIN_RT_R2 = 0.70


def main():
    df = pd.read_csv(DATA / "calibration.csv")
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=0)

    # ---------------------------------------------------------- metric layer
    from autoscaler.metric_estimator import _design

    estimator = MetricEstimator().fit(train_df)
    print(f"metric estimator (R^2 filter at {C.METRIC_R2_THRESHOLD})")
    metric_ok = True
    for m in C.LOAD_METRICS:
        X = _design(
            test_df[f"before_{m}"].to_numpy(),
            test_df["vm_before"].to_numpy(),
            test_df["k"].to_numpy(),
        )
        held_out = r2_score(test_df[f"after_{m}"].to_numpy(), estimator.models[m].predict(X))
        kept = m in estimator.retained
        print(f"  {m:<14} cv {estimator.cv_r2[m]:6.3f}   held-out {held_out:6.3f}   "
              f"{'RETAINED' if kept else 'dropped'}")
        if kept and held_out <= C.METRIC_R2_THRESHOLD:
            metric_ok = False
    print(f"  -> predictor features: {', '.join(estimator.features)}")

    # ------------------------------------------------------------- RT layer
    feats = list(estimator.features)
    tr = rt_training_frame(train_df, estimator.retained)
    te = rt_training_frame(test_df, estimator.retained)
    X_tr, y_tr = tr[feats].to_numpy(), tr[C.TARGET].to_numpy()
    X_te, y_te = te[feats].to_numpy(), te[C.TARGET].to_numpy()

    predictor = EnsembleRTPredictor().fit(X_tr, y_tr)
    pred = predictor.predict(X_te)

    r2 = r2_score(y_te, pred.mean)
    mae = mean_absolute_error(y_te, pred.mean)

    # The decision only ever operates near the SLO, so report accuracy there
    # separately -- a good global R^2 can hide a bad fit in the region that
    # actually drives the scaling choice.
    near = y_te < 2 * C.RT_SLO_MS
    r2_near = r2_score(y_te[near], pred.mean[near])
    mae_near = mean_absolute_error(y_te[near], pred.mean[near])

    print("\nRT ensemble (held-out)")
    print(f"  R^2   all rows        {r2:6.3f}  {'ok' if r2 > MIN_RT_R2 else 'FAIL'}")
    print(f"  MAE   all rows        {mae:6.1f} ms")
    print(f"  R^2   RT < 2x SLO     {r2_near:6.3f}   ({near.sum()} rows)")
    print(f"  MAE   RT < 2x SLO     {mae_near:6.1f} ms")
    epi = predictor.predict(X_te, _epistemic_only=True)
    print(f"  sigma_log (aleatoric) {predictor.sigma_log:6.3f}")
    print(f"  mean epistemic sigma  {epi.std.mean():6.1f} ms")
    print(f"  mean total sigma      {pred.std.mean():6.1f} ms")

    # Coverage: does the one-sided chance constraint hold at its nominal level?
    # If mean + z*sigma is a real 95th percentile, ~95% of actual values should
    # fall below it. This is the number the decision rule silently depends on.
    for label, pr in (("epistemic only", epi), ("epistemic + aleatoric", pred)):
        cov = float(np.mean(y_te <= pr.mean + C.Z_SCORE * pr.std))
        print(f"  coverage of mean+{C.Z_SCORE}s ({label:<21}) {cov:6.1%}  (nominal 95.0%)")

    # ------------------------------------------- does uncertainty mean anything?
    # Binned on the *epistemic* component. Total sigma is dominated by the
    # aleatoric term, which is proportional to the mean by construction, so
    # binning on it would just be re-sorting by predicted response time and
    # would say nothing about whether the ensemble knows where it is unsure.
    order = np.argsort(epi.std)
    abs_err = np.abs(pred.mean - y_te)
    q = np.array_split(order, 5)
    print("\nuncertainty sanity check (quintiles of epistemic sigma)")
    band_sigma, band_err = [], []
    for i, idx in enumerate(q):
        band_sigma.append(epi.std[idx].mean())
        band_err.append(abs_err[idx].mean())
        print(f"  Q{i + 1}  sigma {band_sigma[-1]:8.1f} ms   mean |error| {band_err[-1]:8.1f} ms")
    # The check: does mean error rise as you move to higher-sigma quintiles?
    # A relative confidence signal only has to be ordered, not calibrated.
    monotone = all(b > a for a, b in zip(band_err, band_err[1:]))
    print(f"  error rises monotonically with sigma: {'yes' if monotone else 'no'}")

    # ------------------------------------------------------------------ save
    joblib.dump(estimator, MODELS / "metric_estimator.joblib")
    predictor.save(MODELS / "rt_ensemble.joblib")
    print(f"\nwritten: {MODELS / 'metric_estimator.joblib'}")
    print(f"written: {MODELS / 'rt_ensemble.joblib'}")

    # ----------------------------------------------------------------- plots
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    lim = (30, max(y_te.max(), pred.mean.max()) * 1.05)
    ax.scatter(y_te, pred.mean, s=8, alpha=0.35, color="#2b6cb0", edgecolors="none",
               label="held-out rows")
    ax.plot(lim, lim, color="#c53030", lw=1.2, ls="--", label="y = x")
    ax.axvline(C.RT_SLO_MS, color="#2f855a", lw=1.0, alpha=0.7)
    ax.axhline(C.RT_SLO_MS, color="#2f855a", lw=1.0, alpha=0.7, label=f"SLO {C.RT_SLO_MS:.0f} ms")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("actual response time (ms)")
    ax.set_ylabel("predicted response time (ms)")
    ax.set_title(f"Ensemble mean vs actual\nR² = {r2:.3f}   MAE = {mae:.1f} ms")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_pred_vs_actual.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(band_sigma, band_err, "o-", color="#6b46c1")
    for i, (x, y) in enumerate(zip(band_sigma, band_err)):
        ax.annotate(f"Q{i + 1}", (x, y), textcoords="offset points", xytext=(6, -3), fontsize=8)
    ax.set_xlabel("mean epistemic sigma in quintile (ms)")
    ax.set_ylabel("mean absolute error in quintile (ms)")
    ax.set_title("Does ensemble disagreement track error?")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_uncertainty_check.png", dpi=150)
    print(f"written: {FIGURES / '01_pred_vs_actual.png'}")
    print(f"written: {FIGURES / '02_uncertainty_check.png'}")

    if not (metric_ok and r2 > MIN_RT_R2):
        raise SystemExit("\nSUCCESS CRITERIA NOT MET -- fix the predictor before step 3.")
    print("\nsuccess criteria met.")


if __name__ == "__main__":
    main()
