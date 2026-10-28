package org.uqautoscaler.sim;

/** Measured mean response time against fractions of the (effective) SLO. */
public final class RtThresholdPolicy extends ThresholdPolicy {
    public RtThresholdPolicy(SimConfig cfg) {
        super(cfg, cfg.rtUpperFrac * cfg.effectiveSloMs(), cfg.rtLowerFrac * cfg.effectiveSloMs(), cfg.rtCooldown,
              o -> o.rtMeanMs);
    }
}
