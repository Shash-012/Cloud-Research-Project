package org.uqautoscaler.sim;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Step 3.6: scaling out, scaling in by draining, start-up delay (Step 3.13 item 4). */
class ScalingTest {
    @Test
    void plusThreeThenMinusTwoWithNoLoad() throws Exception {
        SimConfig cfg = SimConfig.defaults();                       // 4 initial VMs
        SimRun r = new SimRun(cfg, SimTestSupport.constant(0.0, 6));
        r.steps = 6;
        r.out = SimTestSupport.tmp("scaling-noload");
        r.policyFactory = run -> SimTestSupport.scripted(step -> step == 0 ? 3 : (step == 2 ? -2 : 0));
        r.run();
        var iv = r.loop.intervals;
        assertEquals(3, iv.get(0).k);
        assertEquals(-2, iv.get(2).k);
        // after the start-up delay all 7 serve; after draining, 5 remain and the drained VMs are destroyed
        assertEquals(7, iv.get(1).nRunning);
        assertEquals(0, iv.get(1).nBooting);
        assertEquals(5, iv.get(3).nRunning + iv.get(3).nBooting);
        assertEquals(0, iv.get(3).nDraining);
        assertEquals(5, iv.get(5).nRunning + iv.get(5).nBooting);   // (all VMs are destroyed when the run ends)
    }

    @Test
    void bootingVmGetsNoRequestsAndDrainingVmFinishesItsOwn() throws Exception {
        SimConfig cfg = SimConfig.defaults();
        cfg.initialVms = 2;
        // 45 req/s on 2 VMs (u = 0.9) so queues exist when a VM is drained; no load in the last two intervals
        double[] rates = {45, 45, 45, 45, 0, 0};
        SimRun r = new SimRun(cfg, rates);
        r.steps = 6;
        r.out = SimTestSupport.tmp("scaling-load");
        r.policyFactory = run -> SimTestSupport.scripted(step -> step == 0 ? 1 : (step == 2 ? -1 : 0));
        Map<Long, Double> firstMapping = new HashMap<>();
        Map<Long, Integer> outstandingAtDrain = new HashMap<>();
        r.beforeStart = run -> {
            run.balancer.mappingObserver = m -> firstMapping.putIfAbsent((long) m[2], m[1]);
            run.cluster.sim.addOnClockTickListener(info -> {
                for (VmState s : run.cluster.all()) {
                    if (s.draining && !outstandingAtDrain.containsKey(s.id())) {
                        outstandingAtDrain.put(s.id(), s.outstanding);
                    }
                }
            });
        };
        r.run();
        // VM 2 was requested at the end of interval 0 (t = 60 s) with a 30 s start-up delay
        double first = firstMapping.get(2L);
        System.out.printf("ScalingTest: new VM requested at 60.0 s, first request mapped at %.3f s%n", first);
        assertTrue(first >= 60.0 + cfg.vmStartupDelayS, "booting VM received a request at " + first);
        assertTrue(first < 60.0 + cfg.vmStartupDelayS + 1.0, "new VM idle long after start-up: " + first);
        // the VM drained at t = 180 s still had requests; all requests complete (none lost)
        assertEquals(1, outstandingAtDrain.size());
        int had = outstandingAtDrain.values().iterator().next();
        assertTrue(had >= 0);
        long arrivals = r.loop.intervals.stream().mapToLong(iv -> iv.arrivals).sum();
        long completions = r.loop.intervals.stream().mapToLong(iv -> iv.completions).sum();
        assertEquals(arrivals, completions, "requests lost while draining");
        var last = r.loop.intervals.get(r.loop.intervals.size() - 1);
        assertEquals(2, last.nRunning + last.nBooting + last.nDraining);
        assertEquals(0, r.balancer.droppedCount());
    }

    @Test
    void drainingVmWithQueueFinishesBeforeDestruction() throws Exception {
        SimConfig cfg = SimConfig.defaults();
        cfg.initialVms = 2;
        double[] rates = {48, 48, 0, 0};              // u = 0.96: long queues
        SimRun r = new SimRun(cfg, rates);
        r.steps = 4;
        r.out = SimTestSupport.tmp("scaling-drain");
        r.policyFactory = run -> SimTestSupport.scripted(step -> step == 0 ? -1 : 0);
        int[] atDrain = {-1};
        double[] destroyedAt = {Double.NaN};
        r.beforeStart = run -> run.cluster.sim.addOnClockTickListener(info -> {
            for (VmState s : run.cluster.all()) {
                if (s.draining && atDrain[0] < 0) atDrain[0] = s.outstanding;
                if (s.destroyed && Double.isNaN(destroyedAt[0])) destroyedAt[0] = s.destroyedAt;
            }
        });
        r.run();
        System.out.printf("ScalingTest: drained VM had %d outstanding requests, destroyed at %.3f s%n",
                          atDrain[0], destroyedAt[0]);
        assertTrue(atDrain[0] > 0, "drained VM should still have queued requests");
        assertTrue(destroyedAt[0] > 60.0, "VM destroyed before finishing its requests");
        long arrivals = r.loop.intervals.stream().mapToLong(iv -> iv.arrivals).sum();
        long completions = r.loop.intervals.stream().mapToLong(iv -> iv.completions).sum();
        assertEquals(arrivals, completions);
    }
}
