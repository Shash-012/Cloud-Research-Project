package org.uqautoscaler.sim;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Step 3.4: the balancer must choose a VM when a request arrives, not when a batch is submitted. */
class ArrivalTimingTest {
    @Test
    void mappingHappensAtPlannedArrivalTime() throws Exception {
        SimConfig cfg = SimConfig.defaults();
        SimRun r = new SimRun(cfg, SimTestSupport.constant(50.0, 3));
        r.steps = 3;
        r.out = SimTestSupport.tmp("arrival-timing");
        List<double[]> mapped = new ArrayList<>();
        List<Double> planned = new ArrayList<>();
        r.beforeStart = run -> run.balancer.mappingObserver = m -> {
            mapped.add(m);
            planned.add(run.arrivals.plannedArrivalOf((long) m[0]));
        };
        r.run();
        assertTrue(mapped.size() > 5000, "expected thousands of requests, got " + mapped.size());
        double maxErr = 0;
        for (int i = 0; i < mapped.size(); i++) {
            maxErr = Math.max(maxErr, Math.abs(mapped.get(i)[1] - planned.get(i)));
        }
        System.out.printf("ArrivalTimingTest: %d requests, max |mapping clock - planned arrival| = %.3g s%n",
                          mapped.size(), maxErr);
        assertEquals(0.0, maxErr, 1e-9);
        // arrivals are spread over the interval, not bunched at its start
        double meanArrivalWithinInterval = mapped.stream().mapToDouble(m -> m[1] % 60.0).average().orElseThrow();
        assertEquals(30.0, meanArrivalWithinInterval, 1.5);
    }
}
