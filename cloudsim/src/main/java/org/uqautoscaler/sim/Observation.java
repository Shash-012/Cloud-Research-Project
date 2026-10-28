package org.uqautoscaler.sim;

import java.util.LinkedHashMap;
import java.util.Map;

/** What a policy sees at the end of an interval. */
public final class Observation {
    public int step;
    public double timeS;
    public int nRunning, nBooting, nDraining;
    public double rtMeanMs;
    public double rate;
    public final Map<String, Double> metrics = new LinkedHashMap<>();

    static Observation of(Telemetry.Interval r) {
        Observation o = new Observation();
        o.step = r.step;
        o.timeS = r.timeS;
        o.nRunning = r.nRunning;
        o.nBooting = r.nBooting;
        o.nDraining = r.nDraining;
        o.rtMeanMs = r.rtMeanMs;
        o.rate = r.rate;
        o.metrics.put("cpu_util", r.cpuUtil);
        o.metrics.put("mem_util", r.memUtil);
        o.metrics.put("net_in_kbps", r.netInKbps);
        o.metrics.put("queue_len", r.queueLen);
        o.metrics.put("throughput_rps", r.throughputRps);
        return o;
    }
}
