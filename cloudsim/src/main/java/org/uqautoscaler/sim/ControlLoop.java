package org.uqautoscaler.sim;

import org.cloudsimplus.listeners.EventInfo;

import java.util.ArrayList;
import java.util.List;

/**
 * Clock-tick listener that closes each interval, logs it, asks the policy and acts (Step 3.9).
 * ArrivalGenerator schedules a marker event at every interval boundary, so a tick happens exactly at each boundary
 * even when no request is in flight.
 */
public final class ControlLoop {
    private final Cluster cluster;
    private final Telemetry telemetry;
    private final EnergyMeter energy;
    private final ArrivalGenerator arrivals;
    private final ScalingActuator actuator;
    private final Policy policy;
    private final int steps;
    private final int forceStep, forceK;
    private final RunLogger logger;
    private double lastT = 0;
    private int closed = 0;
    public final List<Telemetry.Interval> intervals = new ArrayList<>();
    public RuntimeException failure;

    public ControlLoop(Cluster cluster, Telemetry telemetry, EnergyMeter energy, ArrivalGenerator arrivals,
                       ScalingActuator actuator, Policy policy, int steps, int forceStep, int forceK, RunLogger logger) {
        this.cluster = cluster;
        this.telemetry = telemetry;
        this.energy = energy;
        this.arrivals = arrivals;
        this.actuator = actuator;
        this.policy = policy;
        this.steps = steps;
        this.forceStep = forceStep;
        this.forceK = forceK;
        this.logger = logger;
        cluster.sim.addOnClockTickListener(this::onTick);
    }

    private void onTick(EventInfo info) {
        if (failure != null) return;
        try {
            double now = cluster.sim.clock();
            double interval = cluster.cfg.intervalS;
            while (closed < steps && now >= (closed + 1) * interval - 1e-9) {
                double boundary = (closed + 1) * interval;
                integrate(boundary - lastT);
                lastT = boundary;
                closeInterval(closed, now);
                closed++;
            }
            integrate(now - lastT);
            lastT = now;
            actuator.retryPendingDestructions();
        } catch (RuntimeException e) {
            failure = e;
            cluster.sim.abort();
        }
    }

    private void integrate(double dt) {
        telemetry.integrate(dt);
        energy.integrate(dt);
    }

    private void closeInterval(int step, double now) {
        SimConfig cfg = cluster.cfg;
        Telemetry.Interval r = telemetry.close(step, now);
        double[] e = energy.close(cfg.powerModel);
        r.energyWh = e[0];
        r.energyCloudsimWh = e[1];
        r.costUsd = r.vmSeconds / 3600.0 * cfg.pricePerVmHour;

        Observation obs = Observation.of(r);
        int k = policy.decide(obs);
        r.fallback = policy.lastFallback();
        if (step == forceStep) {
            k = forceK;
        }
        r.k = step + 1 < steps ? actuator.apply(k) : 0;
        actuator.retryPendingDestructions();
        intervals.add(r);
        logger.row(r);
        cluster.releaseFinishedCloudlets();
        if (step + 1 < steps) {
            arrivals.scheduleInterval(step + 1);
        }
    }

    public int closedIntervals() {
        return closed;
    }
}
