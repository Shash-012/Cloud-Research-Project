"""Stage-1 input features (Step 2.5) for metric value m, current pool w and action k."""
import numpy as np


def _n(w, k):
    return np.maximum(np.asarray(w, float) + np.asarray(k, float), 1.0)


def paper_features(m, w, k) -> np.ndarray:
    """[m*w/n, m*k/n] (base paper Eq. 1)."""
    m, w, k = (np.asarray(v, float) for v in (m, w, k))
    n = _n(w, k)
    return np.column_stack([m * w / n, m * k / n])


def basis_features(m, w, k) -> np.ndarray:
    """[m, w, k, m*w/n]."""
    m, w, k = (np.asarray(v, float) for v in (m, w, k))
    n = _n(w, k)
    m, w, k = np.broadcast_arrays(m, w, k)
    return np.column_stack([m, w, k, m * w / n])


FEATURES = {"paper": paper_features, "basis": basis_features}


def transition_inputs(df, metric):
    """(m_before, w, k_effective) from a transition table.

    k_effective = n_after - n_before: the action after clamping to [vm_min, vm_max], so the
    features see the pool that actually served. The controller only scores in-range actions,
    where k_effective == k.
    """
    m = df[f"{metric}_before"].to_numpy(float)
    w = df["n_before"].to_numpy(float)
    k = (df["n_after"] - df["n_before"]).to_numpy(float)
    return m, w, k
