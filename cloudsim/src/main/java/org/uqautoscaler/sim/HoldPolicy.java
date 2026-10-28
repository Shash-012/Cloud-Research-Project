package org.uqautoscaler.sim;

/** Always k = 0; used by tests and the counterfactual check (with --force-action). */
public final class HoldPolicy implements Policy {
    @Override
    public int decide(Observation obs) {
        return 0;
    }
}
