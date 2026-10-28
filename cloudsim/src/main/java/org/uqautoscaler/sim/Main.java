package org.uqautoscaler.sim;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Entry point. Arguments: --config cfg.json --workload-csv w.csv --seed N --policy NAME --out DIR [--port P]
 * [--run-id ID] [--workload-name NAME] [--force-action STEP:K] [--steps N] [--initial-vms N] [--energy-crosscheck]
 * [--remote] (serve even a Java policy from Python, for twin checks).
 * Java policies: hold, random_target, cpu_threshold, rt_threshold. Any other name is served by the Python bridge.
 * Without arguments it prints the Phase 0 smoke-test line.
 */
public class Main {
    public static void main(String[] args) {
        if (args.length == 0) {
            org.cloudsimplus.core.CloudSimPlus sim = new org.cloudsimplus.core.CloudSimPlus();
            System.out.println("CloudSim Plus engine created; clock = " + sim.clock());
            return;
        }
        Map<String, String> a = parse(args);
        try {
            SimConfig cfg = a.containsKey("config") ? SimConfig.fromJson(Path.of(a.get("config"))) : SimConfig.defaults();
            if (a.containsKey("initial-vms")) cfg.initialVms = Integer.parseInt(a.get("initial-vms"));
            double[] rates = readRates(Path.of(a.get("workload-csv")));
            SimRun r = new SimRun(cfg, rates);
            r.seed = Long.parseLong(a.getOrDefault("seed", "1"));
            r.policyName = a.getOrDefault("policy", "hold");
            r.workloadName = a.getOrDefault("workload-name", Path.of(a.get("workload-csv")).getFileName().toString());
            r.runId = a.getOrDefault("run-id", r.policyName + "_" + r.workloadName + "_s" + r.seed);
            r.out = Path.of(a.getOrDefault("out", "."));
            if (a.containsKey("steps")) r.steps = Integer.parseInt(a.get("steps"));
            if (a.containsKey("force-action")) {
                String[] f = a.get("force-action").split(":");
                r.forceStep = Integer.parseInt(f[0]);
                r.forceK = Integer.parseInt(f[1]);
            }
            r.energyCrossCheck = a.containsKey("energy-crosscheck");
            int port = Integer.parseInt(a.getOrDefault("port", Integer.toString(cfg.bridgePort)));
            List<String> javaPolicies = List.of("hold", "random_target", "cpu_threshold", "rt_threshold");
            if (!javaPolicies.contains(r.policyName) || a.containsKey("remote")) {
                r.policyFactory = run -> new RemotePolicy(cfg, cfg.bridgeHost, port, run.runId, run.policyName, run.seed);
            }
            Map<String, Object> s = r.run();
            System.out.println("run complete: " + s);
        } catch (IllegalStateException e) {
            System.err.println("ERROR: " + e.getMessage());
            System.exit(3);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(2);
        }
    }

    static Map<String, String> parse(String[] args) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (!args[i].startsWith("--")) continue;
            String key = args[i].substring(2);
            if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                m.put(key, args[++i]);
            } else {
                m.put(key, "true");
            }
        }
        return m;
    }

    /** CSV with header `step,rate`, rates in unscaled requests per second. */
    public static double[] readRates(Path csv) throws IOException {
        List<String> lines = Files.readAllLines(csv);
        List<Double> rates = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) continue;
            rates.add(Double.parseDouble(line.split(",")[1]));
        }
        return rates.stream().mapToDouble(Double::doubleValue).toArray();
    }
}
