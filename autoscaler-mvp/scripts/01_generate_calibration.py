"""Step 1: build the synthetic environment's calibration dataset.

Writes data/calibration.csv and a figure of the workload timeline.

Success criterion: the workload visibly contains a ramp and a burst, and the
calibration set covers the whole VM range at a spread of request rates.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _bootstrap import DATA, FIGURES  # noqa: E402

from autoscaler import config as C  # noqa: E402
from autoscaler.calibration import collect  # noqa: E402
from autoscaler.environment import generate_workload  # noqa: E402


def main():
    workload = generate_workload()
    df = collect()

    out = DATA / "calibration.csv"
    df.to_csv(out, index=False)

    print(f"calibration rows : {len(df)}")
    print(f"vm_before range  : {df['vm_before'].min():.0f} - {df['vm_before'].max():.0f}")
    print(f"request rate     : {df['request_rate'].min():.0f} - {df['request_rate'].max():.0f} req/s")
    print(f"response time    : {df[C.TARGET].min():.0f} - {df[C.TARGET].max():.0f} ms")
    print(f"rows over SLO    : {(df[C.TARGET] > C.RT_SLO_MS).mean():.1%}")
    print(f"written          : {out}")

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(workload, lw=1.0, color="#2b6cb0")
    ax.axvspan(C.RAMP_START, C.RAMP_END, alpha=0.10, color="green", label="ramp")
    ax.axvspan(
        C.SPIKE_CENTER - 2 * C.SPIKE_WIDTH,
        C.SPIKE_CENTER + 2 * C.SPIKE_WIDTH,
        alpha=0.10,
        color="red",
        label="burst",
    )
    ax.set_xlabel("control step (1 min)")
    ax.set_ylabel("requests / s")
    ax.set_title("Synthetic workload")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "00_workload.png", dpi=150)
    print(f"written          : {FIGURES / '00_workload.png'}")


if __name__ == "__main__":
    main()
