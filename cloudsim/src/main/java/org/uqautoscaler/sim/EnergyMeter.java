package org.uqautoscaler.sim;

import org.cloudsimplus.hosts.Host;

/**
 * Energy per interval from each host's busy fraction and a configurable power model (Step 3.8):
 * E_t = sum over hosts with a VM of [P_idle + (P_max - P_idle) h(u)] * dt / 3600 Wh, u = busy / alive time of the
 * host's VM in the interval (0 while booting). Also integrates CloudSim Plus's own host power model when the
 * cluster was built with one, for the cross-check.
 */
public final class EnergyMeter {
    private final Cluster cluster;
    private final boolean crossCheck;
    private double cloudsimJoules;

    public EnergyMeter(Cluster cluster, boolean crossCheck) {
        this.cluster = cluster;
        this.crossCheck = crossCheck;
    }

    public static double h(String model, double u) {
        u = Math.max(0.0, Math.min(1.0, u));
        return switch (model) {
            case "linear" -> u;
            case "square" -> u * u;
            case "cubic" -> u * u * u;
            case "sqrt" -> Math.sqrt(u);
            default -> throw new IllegalArgumentException("unknown power model " + model);
        };
    }

    public static double power(SimConfig cfg, String model, double u) {
        return cfg.pIdleW + (cfg.pMaxW - cfg.pIdleW) * h(model, u);
    }

    /** CloudSim Plus host power, integrated on clock ticks (state is constant between ticks). */
    void integrate(double dt) {
        if (!crossCheck || dt <= 0) return;
        for (VmState s : cluster.all()) {
            if (!s.isAlive() || !s.vm.isCreated()) continue;
            Host host = s.vm.getHost();
            cloudsimJoules += host.getPowerModel().getPower() * dt;
        }
    }

    /** Close the interval: energy in Wh with the given model; resets per-VM accumulators. */
    public double[] close(String model) {
        double wh = 0;
        for (VmState s : cluster.all()) {
            if (s.aliveS > 0) {
                double u = s.aliveS > 0 ? s.busyS / s.aliveS : 0.0;
                wh += s.aliveS * power(cluster.cfg, model, u) / 3600.0;
            }
            s.resetInterval();
        }
        double cs = cloudsimJoules / 3600.0;
        cloudsimJoules = 0;
        return new double[]{wh, cs};
    }
}
