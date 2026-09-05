"""Central constants for the MVP.

Every magic number in the prototype lives here so that a reviewer can see the
whole parameterisation of the synthetic environment in one place.

NOTE (scope): this is a *synthetic* environment. None of these values are
calibrated against a real cloud provider or against CloudSim Plus. They are
chosen so the simulated system spends time in all three interesting regimes:
comfortably under the SLO, near the SLO boundary, and briefly infeasible.
"""

# ---------------------------------------------------------------- environment
VM_MIN = 1                  # never scale below this
VM_MAX = 20                 # hard ceiling on the pool
SERVICE_RATE_PER_VM = 25.0  # requests/second a single VM can serve
BASE_LATENCY_MS = 40.0      # zero-queueing service latency

# Slowly varying "request complexity" multiplier on service demand. This is an
# *unobserved* driver of response time: the predictor never sees it. It exists
# so that the RT surface is not perfectly learnable, which is what gives the
# ensemble's uncertainty estimate something real to represent.
COMPLEXITY_AMPLITUDE = 0.12
COMPLEXITY_PERIOD = 260     # in control steps
COMPLEXITY_NOISE_STD = 0.03

RT_NOISE_FRAC = 0.05        # multiplicative observation noise on response time

# --------------------------------------------------------------------- workload
N_STEPS = 1440              # 1440 control steps == 24h at 1 step/minute
BASE_RATE = 120.0           # requests/second, mean level of the daily cycle
DAILY_AMPLITUDE = 60.0
RAMP_START, RAMP_END = 300, 800
RAMP_HEIGHT = 150.0
SPIKE_CENTER, SPIKE_WIDTH, SPIKE_HEIGHT = 1050, 40.0, 180.0
RATE_NOISE_STD = 8.0
RATE_MIN, RATE_MAX = 30.0, 460.0

# ---------------------------------------------------------------------- SLO
RT_SLO_MS = 250.0           # latency SLO the autoscaler must respect
Z_SCORE = 1.65              # one-sided ~95th percentile: mean + z*std <= SLO

# ------------------------------------------------------------- cost / energy
CONTROL_INTERVAL_S = 60.0
DT_HOURS = CONTROL_INTERVAL_S / 3600.0

PRICE_PER_VM_HOUR = 0.10    # USD; flat, homogeneous VMs

# Utilisation-aware power model, standing in for CloudSim's PowerModel family.
# Per VM:   P(u) = P_IDLE + (P_MAX - P_IDLE) * u**ALPHA,   u = lambda / (n * mu)
#
# Total pool power is then  n*P_IDLE + dP*lambda**2 / (mu**2 * n), which is
# U-shaped in n with its minimum at  n* = (lambda/mu) * sqrt(dP / P_IDLE).
# More VMs means more idle draw but each runs at lower utilisation, and with
# ALPHA > 1 the dynamic term falls faster than the idle term grows. That
# non-monotonicity is what makes (cost, energy) a genuine trade-off instead of
# two names for the same number.
#
# The ratio dP / P_IDLE decides whether the trade-off is visible at all. The
# SLO admits pools of at least  n_slo = lambda / (mu * u_slo)  where
# u_slo = 1 - BASE_LATENCY / RT_SLO. A front with more than one point on it
# requires n* > n_slo, i.e. sqrt(dP / P_IDLE) > 1 / u_slo. At u_slo = 0.84 that
# means dP / P_IDLE > 1.42. The values below give a ratio of 4.0 (n* is ~68%
# above n_slo), which assumes fairly energy-proportional hardware: idle draw is
# 20% of peak. On less proportional hardware the feasible front shrinks toward
# the single cheapest action -- worth reporting as a sensitivity result rather
# than hiding, since it bounds when the Pareto stage earns its keep.
P_IDLE_W = 80.0
P_MAX_W = 400.0
POWER_ALPHA = 2.0           # square power model, as in CloudSim's PowerModelSquare

# ------------------------------------------------------------------ decision
K_CANDIDATES = (-2, -1, 0, 1, 2)

# -------------------------------------------------------------- calibration
CALIBRATION_ROWS = 4000
CALIBRATION_SEED = 7
WORKLOAD_SEED = 11
SIM_SEED = 23

# ------------------------------------------------------------------ ensemble
N_ENSEMBLE = 5
HIDDEN_LAYERS = (64, 32)
MAX_ITER = 600
ENSEMBLE_BASE_SEED = 100

# Every metric the synthetic app emits. Which of these actually reach the RT
# predictor is decided at fit time by the estimator's R^2 filter, not here.
LOAD_METRICS = ("cpu_util", "mem_util", "net_in_kbps", "disk_iops", "queue_len")

# The base paper's second metric-selection criterion: keep a metric only if a
# linear model can explain its post-scaling value well. Metrics that fail are
# ones whose post-scaling value cannot be estimated reliably, so feeding them to
# the RT network at decision time would mean feeding it noise.
METRIC_R2_THRESHOLD = 0.90

TARGET = "response_time_ms"


def features_for(metrics) -> tuple:
    """Feature order used by every module. vm_count and request_rate always lead."""
    return ("vm_count", "request_rate") + tuple(metrics)


# Full feature set, before selection. Used for defaults and tests.
FEATURES = features_for(LOAD_METRICS)
