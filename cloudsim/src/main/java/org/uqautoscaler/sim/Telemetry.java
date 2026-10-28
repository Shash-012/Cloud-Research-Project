package org.uqautoscaler.sim;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Per-interval metrics (Step 3.7). Time-based quantities (busy time, outstanding requests, VM-seconds) are
 * integrated on every clock tick: between two ticks no event happens, so the state seen at a tick is exactly the
 * state of the elapsed slice and the integral is exact. This is equivalent to splitting each cloudlet's
 * [start, finish] busy period across interval boundaries, including still-running cloudlets.
 */
public final class Telemetry {
    private final Cluster cluster;
    private final SplittableRandom memRng;
    private long arrivals, completions;
    private double rtSumMs;
    private double[] rts = new double[4096];
    private int nRts;
    private double servingS, busyS, outstandingS, vmAliveS;

    public Telemetry(Cluster cluster, long seed) {
        this.cluster = cluster;
        this.memRng = new SplittableRandom(seed * 1_000_003L + 47);
    }

    void onArrival() {
        arrivals++;
    }

    void onCompletion(double rtMs) {
        completions++;
        rtSumMs += rtMs;
        if (nRts == rts.length) rts = Arrays.copyOf(rts, rts.length * 2);
        rts[nRts++] = rtMs;
    }

    /** Integrate the state of the slice that just elapsed (dt seconds). */
    void integrate(double dt) {
        if (dt <= 0) return;
        for (VmState s : cluster.all()) {
            if (!s.isAlive()) continue;
            s.aliveS += dt;
            vmAliveS += dt;
            if (s.isServing()) {
                s.servingS += dt;
                servingS += dt;
                outstandingS += s.outstanding * dt;
                if (s.isBusy()) {
                    s.busyS += dt;
                    busyS += dt;
                }
            }
        }
    }

    public Interval close(int step, double timeS) {
        SimConfig cfg = cluster.cfg;
        Interval r = new Interval();
        r.step = step;
        r.timeS = timeS;
        r.arrivals = arrivals;
        r.completions = completions;
        r.rate = arrivals / cfg.intervalS;
        r.throughputRps = completions / cfg.intervalS;
        r.netInKbps = arrivals * cfg.requestKb * 8.0 / cfg.intervalS;
        r.rtMeanMs = completions > 0 ? rtSumMs / completions : Double.NaN;
        r.rtP95Ms = percentile(rts, nRts, 0.95);
        r.cpuUtil = servingS > 0 ? 100.0 * busyS / servingS : 0.0;
        r.queueLen = servingS > 0 ? outstandingS / servingS : 0.0;
        double mem = cfg.memOffset + cfg.memSlope * r.cpuUtil / 100.0 + cfg.memNoiseStd * memRng.nextGaussian();
        r.memUtil = Math.max(0.0, Math.min(100.0, mem));
        r.vmSeconds = vmAliveS;
        r.nRunning = cluster.nRunning();
        r.nBooting = cluster.nBooting();
        r.nDraining = cluster.nDraining();
        r.sloViolated = completions > 0 && r.rtMeanMs > cfg.effectiveSloMs();
        arrivals = completions = 0;
        rtSumMs = 0;
        nRts = 0;
        servingS = busyS = outstandingS = vmAliveS = 0;
        return r;
    }

    /** Linear interpolation between order statistics (numpy's default). */
    static double percentile(double[] a, int n, double q) {
        if (n == 0) return Double.NaN;
        double[] s = Arrays.copyOf(a, n);
        Arrays.sort(s);
        double pos = q * (n - 1);
        int lo = (int) Math.floor(pos);
        int hi = Math.min(n - 1, lo + 1);
        return s[lo] + (pos - lo) * (s[hi] - s[lo]);
    }

    /** One closed interval. */
    public static final class Interval {
        public int step;
        public double timeS, rate, rtMeanMs, rtP95Ms, cpuUtil, memUtil, netInKbps, queueLen, throughputRps;
        public long arrivals, completions;
        public int nRunning, nBooting, nDraining;
        public boolean sloViolated;
        public double vmSeconds, costUsd, energyWh, energyCloudsimWh;
        public int k;
        public boolean fallback;
    }
}
