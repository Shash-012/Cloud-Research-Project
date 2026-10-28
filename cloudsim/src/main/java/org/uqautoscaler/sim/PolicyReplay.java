package org.uqautoscaler.sim;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Replays recorded observations (JSON lines) through a Java threshold policy and prints one k per line.
 * Used by the Python test that checks the Java baselines decide like their Python twins (Step 3.10).
 * Usage: PolicyReplay cfg.json obs.jsonl cpu_threshold|rt_threshold
 */
public final class PolicyReplay {
    public static void main(String[] args) throws Exception {
        SimConfig cfg = SimConfig.fromJson(Path.of(args[0]));
        SimRun r = new SimRun(cfg, new double[0]);
        Policy p = SimRun.makePolicy(r, args[2]);
        ObjectMapper json = new ObjectMapper();
        StringBuilder sb = new StringBuilder();
        for (String line : Files.readAllLines(Path.of(args[1]))) {
            if (line.isBlank()) continue;
            JsonNode n = json.readTree(line);
            Observation o = new Observation();
            o.step = n.path("step").asInt();
            o.nRunning = n.path("n_running").asInt();
            o.nBooting = n.path("n_booting").asInt();
            o.rtMeanMs = n.path("rt_mean_ms").isNull() ? Double.NaN : n.path("rt_mean_ms").asDouble();
            o.rate = n.path("rate").asDouble();
            n.path("metrics").fields().forEachRemaining(e -> o.metrics.put(e.getKey(), e.getValue().asDouble()));
            sb.append(p.decide(o)).append('\n');
        }
        System.out.print(sb);
    }
}
