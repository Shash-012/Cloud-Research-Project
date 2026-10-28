package org.uqautoscaler.sim;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** Writes the run log (CSV, one row per interval) and the summary (JSON). */
public final class RunLogger implements AutoCloseable {
    public static final String HEADER = String.join(",",
        "run_id", "policy", "workload", "seed", "step", "time_s", "rate", "n_running", "n_booting", "n_draining",
        "k", "fallback", "rt_mean_ms", "rt_p95_ms", "slo_violated", "cost_usd", "energy_wh", "cpu_util", "mem_util",
        "net_in_kbps", "queue_len", "throughput_rps", "arrivals", "completions", "capacity_scale",
        "energy_cloudsim_wh");
    private final BufferedWriter w;
    private final String prefix;
    private final double scale;

    public RunLogger(Path out, String runId, String policy, String workload, long seed, double scale) {
        try {
            Files.createDirectories(out);
            w = Files.newBufferedWriter(out.resolve("run_log.csv"), StandardCharsets.UTF_8);
            w.write(HEADER);
            w.write('\n');
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        this.prefix = runId + "," + policy + "," + workload + "," + seed + ",";
        this.scale = scale;
    }

    private static String num(double v) {
        return Double.isFinite(v) ? Double.toString(v) : "";
    }

    public void row(Telemetry.Interval r) {
        String line = prefix + r.step + "," + num(r.timeS) + "," + num(r.rate) + "," + r.nRunning + "," + r.nBooting
            + "," + r.nDraining + "," + r.k + "," + r.fallback + "," + num(r.rtMeanMs) + "," + num(r.rtP95Ms) + ","
            + r.sloViolated + "," + num(r.costUsd) + "," + num(r.energyWh) + "," + num(r.cpuUtil) + ","
            + num(r.memUtil) + "," + num(r.netInKbps) + "," + num(r.queueLen) + "," + num(r.throughputRps) + ","
            + r.arrivals + "," + r.completions + "," + num(scale) + "," + num(r.energyCloudsimWh);
        try {
            w.write(line);
            w.write('\n');
            w.flush();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static void writeSummary(Path out, Map<String, Object> summary) throws IOException {
        new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT)
            .writeValue(out.resolve("summary.json").toFile(), summary);
    }

    @Override
    public void close() throws IOException {
        w.close();
    }
}
