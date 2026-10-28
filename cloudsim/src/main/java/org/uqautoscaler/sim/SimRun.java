package org.uqautoscaler.sim;

import ch.qos.logback.classic.Level;
import org.cloudsimplus.util.Log;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/** Builds the whole simulation for one run and executes it. Used by Main and by the JUnit tests. */
public final class SimRun {
    public String runId = "run";
    public String policyName = "hold";
    public String workloadName = "custom";
    public long seed = 1;
    public int steps;
    public int forceStep = -1, forceK = 0;
    public boolean energyCrossCheck = false;
    public Path out;
    public Function<SimRun, Policy> policyFactory;
    public final SimConfig cfg;
    public final double[] rates;

    // available after run() starts
    public Cluster cluster;
    public LoadBalancer balancer;
    public ArrivalGenerator arrivals;
    public Telemetry telemetry;
    public ControlLoop loop;
    public double wallSeconds;
    /** Hook for tests: called after the cluster is built, before the simulation starts. */
    public java.util.function.Consumer<SimRun> beforeStart;

    public SimRun(SimConfig cfg, double[] rates) {
        this.cfg = cfg;
        this.rates = rates;
        this.steps = cfg.stepsPerRun;
    }

    public static Policy makePolicy(SimRun r, String name) {
        return switch (name) {
            case "hold" -> new HoldPolicy();
            case "random_target" -> new RandomTargetPolicy(r.cfg, r.seed);
            case "cpu_threshold" -> new CpuThresholdPolicy(r.cfg);
            case "rt_threshold" -> new RtThresholdPolicy(r.cfg);
            default -> throw new IllegalArgumentException("unknown Java policy " + name);
        };
    }

    public Map<String, Object> run() throws IOException {
        // per-event console logging would slow a run with millions of cloudlets dramatically
        Log.setLevel(Level.WARN);
        long t0 = System.nanoTime();
        cluster = new Cluster(cfg, energyCrossCheck);
        telemetry = new Telemetry(cluster, seed);
        arrivals = new ArrivalGenerator(cluster, rates, seed, telemetry);
        balancer = new LoadBalancer(cluster);
        ScalingActuator actuator = new ScalingActuator(cluster);
        EnergyMeter energy = new EnergyMeter(cluster, energyCrossCheck);
        Policy policy = policyFactory != null ? policyFactory.apply(this) : makePolicy(this, policyName);
        Map<String, Object> summary = new LinkedHashMap<>();
        try (RunLogger logger = new RunLogger(out, runId, policyName, workloadName, seed, cfg.capacityScale)) {
            loop = new ControlLoop(cluster, telemetry, energy, arrivals, actuator, policy, steps, forceStep, forceK,
                                   logger);
            if (beforeStart != null) beforeStart.accept(this);
            cluster.sim.terminateAt(steps * cfg.intervalS);
            cluster.sim.start();
        }
        wallSeconds = (System.nanoTime() - t0) / 1e9;
        List<Telemetry.Interval> iv = loop.intervals;
        summary.put("run_id", runId);
        summary.put("policy", policyName);
        summary.put("workload", workloadName);
        summary.put("seed", seed);
        summary.put("steps", loop.closedIntervals());
        summary.put("requests", arrivals.generated());
        summary.put("dropped_requests", balancer.droppedCount());
        summary.put("capacity_scale", cfg.capacityScale);
        summary.put("energy_wh", iv.stream().mapToDouble(r -> r.energyWh).sum());
        summary.put("energy_cloudsim_wh", iv.stream().mapToDouble(r -> r.energyCloudsimWh).sum());
        summary.put("cost_usd", iv.stream().mapToDouble(r -> r.costUsd).sum());
        summary.put("wall_clock_s", wallSeconds);
        summary.put("status", loop.failure == null ? "ok" : "error: " + loop.failure.getMessage());
        policy.close(summary);
        if (out != null) {
            RunLogger.writeSummary(out, summary);
        }
        if (loop.failure != null) {
            throw loop.failure;
        }
        return summary;
    }
}
