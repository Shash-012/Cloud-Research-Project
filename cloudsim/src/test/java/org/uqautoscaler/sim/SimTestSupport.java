package org.uqautoscaler.sim;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntUnaryOperator;

final class SimTestSupport {
    private SimTestSupport() {
    }

    static double[] constant(double rate, int steps) {
        double[] r = new double[steps];
        Arrays.fill(r, rate);
        return r;
    }

    static Path tmp(String name) throws Exception {
        Path p = Path.of("target", "test-runs", name);
        Files.createDirectories(p);
        return p;
    }

    /** A policy that returns script(step). */
    static Policy scripted(IntUnaryOperator script) {
        return obs -> script.applyAsInt(obs.step);
    }

    /** Request-weighted mean response time over intervals with completions. */
    static double meanRt(List<Telemetry.Interval> iv, int fromStep) {
        double sum = 0;
        long n = 0;
        for (Telemetry.Interval r : iv) {
            if (r.step < fromStep || r.completions == 0) continue;
            sum += r.rtMeanMs * r.completions;
            n += r.completions;
        }
        return sum / n;
    }
}
