package org.uqautoscaler.sim;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Step 3.13: determinism, queueing sanity, utilisation, energy cross-check. */
class ValidationTest {
    @Test
    void determinismByteIdenticalRunLogs() throws Exception {
        byte[][] logs = new byte[3][];
        for (int i = 0; i < 3; i++) {
            SimConfig cfg = SimConfig.defaults();
            double[] rates = new double[30];
            for (int t = 0; t < 30; t++) rates[t] = 80 + 60 * Math.sin(t / 5.0);
            SimRun r = new SimRun(cfg, rates);
            r.steps = 30;
            r.seed = i < 2 ? 3 : 4;
            r.policyName = "random_target";
            r.out = SimTestSupport.tmp("determinism-" + i);
            r.run();
            logs[i] = Files.readAllBytes(r.out.resolve("run_log.csv"));
        }
        assertArrayEquals(logs[0], logs[1], "same seed must give byte-identical run logs");
        assertTrue(!Arrays.equals(logs[0], logs[2]), "a different seed should change the log");
    }

    /** Run at full scale and at the capacity_scale used for experiments (Step 3.1, "Run time"). */
    @ParameterizedTest
    @ValueSource(doubles = {1.0, 0.25})
    void mm1MeanResponseTimeWithinFivePercent(double scale) throws Exception {
        SimConfig cfg = SimConfig.defaults();
        cfg.capacityScale = scale;
        cfg.initialVms = 1;
        // lambda = 15 req/s (unscaled) for 6 h; with scale s rates are x s and service times / s
        int steps = scale == 1.0 ? 360 : 1440;              // keep ~324k requests at s = 0.25
        SimRun r = new SimRun(cfg, SimTestSupport.constant(15.0, steps));
        r.steps = steps;
        r.out = SimTestSupport.tmp("mm1-" + scale);
        r.run();
        double mean = SimTestSupport.meanRt(r.loop.intervals, 0);
        double expected = 1000.0 / (25.0 * scale - 15.0 * scale);
        System.out.printf("M/M/1 (s=%.2f): mean response time %.2f ms vs %.2f ms (%.2f%%)%n", scale, mean, expected,
                          100 * (mean - expected) / expected);
        assertEquals(expected, mean, 0.05 * expected);
    }

    @ParameterizedTest
    @ValueSource(doubles = {1.0, 0.25})
    void utilisationWithinTwoPoints(double scale) throws Exception {
        for (double[] c : new double[][]{{150.0, 8}, {60.0, 4}, {300.0, 20}}) {
            SimConfig cfg = SimConfig.defaults();
            cfg.capacityScale = scale;
            cfg.initialVms = (int) c[1];
            SimRun r = new SimRun(cfg, SimTestSupport.constant(c[0], 60));
            r.steps = 60;
            r.out = SimTestSupport.tmp("util-" + (int) c[0]);
            r.run();
            double measuredRate = r.loop.intervals.stream().mapToDouble(iv -> iv.rate).average().orElseThrow();
            double cpu = r.loop.intervals.stream().mapToDouble(iv -> iv.cpuUtil).average().orElseThrow();
            double expected = 100.0 * c[0] / (c[1] * 25.0);
            System.out.printf("utilisation (s=%.2f): lambda=%.0f n=%d cpu %.2f%% vs %.2f%% (measured rate %.2f)%n",
                              scale, c[0], (int) c[1], cpu, expected, measuredRate);
            assertEquals(expected, cpu, 2.0);
        }
    }

    @ParameterizedTest
    @ValueSource(doubles = {1.0, 0.25})
    void energyMatchesCloudSimLinearPowerModel(double scale) throws Exception {
        SimConfig cfg = SimConfig.defaults();
        cfg.capacityScale = scale;
        cfg.powerModel = "linear";
        cfg.initialVms = 6;
        double[] rates = new double[30];
        for (int t = 0; t < 30; t++) rates[t] = 60 + 50 * Math.sin(t / 4.0);
        SimRun r = new SimRun(cfg, rates);
        r.steps = 30;
        r.policyName = "cpu_threshold";                     // includes scale-out, booting and draining VMs
        r.energyCrossCheck = true;
        r.out = SimTestSupport.tmp("energy-" + scale);
        Map<String, Object> s = r.run();
        double ours = (double) s.get("energy_wh");
        double cloudsim = (double) s.get("energy_cloudsim_wh");
        System.out.printf("energy cross-check (s=%.2f): EnergyMeter %.4f Wh vs CloudSim Plus linear %.4f Wh (%.4f%%)%n",
                          scale, ours, cloudsim, 100 * (ours - cloudsim) / cloudsim);
        assertTrue(cloudsim > 0);
        assertEquals(cloudsim, ours, 0.01 * cloudsim);
    }

    @Test
    void runLogHasOneRowPerStep() throws Exception {
        SimConfig cfg = SimConfig.defaults();
        SimRun r = new SimRun(cfg, SimTestSupport.constant(100.0, 12));
        r.steps = 12;
        r.out = SimTestSupport.tmp("rows");
        r.run();
        long rows = Files.readAllLines(r.out.resolve("run_log.csv")).size() - 1;
        assertEquals(12, rows);
        assertEquals(12, r.loop.intervals.get(11).step + 1);
        Path summary = r.out.resolve("summary.json");
        assertTrue(Files.exists(summary));
    }
}
