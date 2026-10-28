package org.uqautoscaler.sim;

public final class CpuThresholdPolicy extends ThresholdPolicy {
    public CpuThresholdPolicy(SimConfig cfg) {
        super(cfg, cfg.cpuUpper, cfg.cpuLower, cfg.cpuCooldown, o -> o.metrics.get("cpu_util"));
    }
}
