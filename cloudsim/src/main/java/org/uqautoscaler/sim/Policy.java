package org.uqautoscaler.sim;

import java.util.Map;

/** A decision rule: an action k for the observation of the interval that just closed. */
public interface Policy {
    int decide(Observation obs);

    /** Whether the last decision was a fallback (no feasible candidate); only remote policies set it. */
    default boolean lastFallback() {
        return false;
    }

    /** Called once after the run with the run summary. */
    default void close(Map<String, Object> summary) {
    }

    /** Clamp k to the action range and the pool limits; 0 if the pool would leave [vm_min, vm_max]. */
    static int valid(SimConfig cfg, int w, int k) {
        if (k < cfg.kMin || k > cfg.kMax || w + k < cfg.vmMin || w + k > cfg.vmMax) {
            return 0;
        }
        return k;
    }
}
