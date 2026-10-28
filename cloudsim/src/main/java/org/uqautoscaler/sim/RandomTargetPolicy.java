package org.uqautoscaler.sim;

import java.util.SplittableRandom;

/**
 * Calibration-data policy (Step 3.14): draw a target utilisation u* ~ U(u_min, u_max), take the pool
 * n* = ceil(lambda_last / (u* mu)) that would reach it at the last interval's rate, set k = clip(n* - w, k_min, k_max)
 * and hold the resulting pool for `hold_intervals` intervals. Targets above 1 deliberately visit overload.
 */
public final class RandomTargetPolicy implements Policy {
    private final SimConfig cfg;
    private final SplittableRandom rng;
    private int hold = 0;

    public RandomTargetPolicy(SimConfig cfg, long seed) {
        this.cfg = cfg;
        this.rng = new SplittableRandom(seed * 1_000_003L + 71);
    }

    @Override
    public int decide(Observation obs) {
        if (hold > 0) {
            hold--;
            return 0;
        }
        double u = cfg.rtUMin + (cfg.rtUMax - cfg.rtUMin) * rng.nextDouble();
        int w = obs.nRunning + obs.nBooting;
        int nStar = (int) Math.ceil(obs.rate / (u * cfg.muPerVm()));
        int k = Math.max(cfg.kMin, Math.min(cfg.kMax, nStar - w));
        k = Math.max(cfg.vmMin - w, Math.min(cfg.vmMax - w, k));
        hold = cfg.rtHoldIntervals - 1;
        return k;
    }
}
