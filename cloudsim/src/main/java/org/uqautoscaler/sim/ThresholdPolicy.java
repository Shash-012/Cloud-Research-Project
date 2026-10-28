package org.uqautoscaler.sim;

import java.util.function.ToDoubleFunction;

/**
 * Same rule and parameters as the Python baselines (decision/policies.py): +1 above `upper`, -1 below `lower`,
 * nothing during the cooldown that follows an action; an action leaving the pool limits becomes 0.
 */
public class ThresholdPolicy implements Policy {
    private final SimConfig cfg;
    private final double upper, lower;
    private final int cooldownSteps;
    private final ToDoubleFunction<Observation> signal;
    private int cooldown = 0;

    ThresholdPolicy(SimConfig cfg, double upper, double lower, int cooldownSteps, ToDoubleFunction<Observation> signal) {
        this.cfg = cfg;
        this.upper = upper;
        this.lower = lower;
        this.cooldownSteps = cooldownSteps;
        this.signal = signal;
    }

    @Override
    public int decide(Observation obs) {
        if (cooldown > 0) {
            cooldown--;
            return 0;
        }
        double x = signal.applyAsDouble(obs);
        int k = x > upper ? 1 : (x < lower ? -1 : 0);
        k = Policy.valid(cfg, obs.nRunning + obs.nBooting, k);
        if (k != 0) {
            cooldown = cooldownSteps;
        }
        return k;
    }
}
