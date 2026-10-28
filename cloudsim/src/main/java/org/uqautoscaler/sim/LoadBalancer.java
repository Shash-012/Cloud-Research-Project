package org.uqautoscaler.sim;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.vms.Vm;

import java.util.HashSet;
import java.util.Set;
import java.util.function.Consumer;

/**
 * The broker's VM mapper (Step 3.5): only created, started-up, non-draining VMs are eligible; the one with the
 * fewest outstanding requests wins, ties to the lowest VM id. `round_robin` is a sensitivity option.
 */
public final class LoadBalancer {
    private final Cluster cluster;
    private final boolean roundRobin;
    private long lastRoundRobinId = -1;
    private final Set<Long> dropped = new HashSet<>();
    /** Observer for tests: (cloudlet, clock at mapping time). */
    public Consumer<double[]> mappingObserver;

    public LoadBalancer(Cluster cluster) {
        this.cluster = cluster;
        this.roundRobin = "round_robin".equals(cluster.cfg.balancer);
        cluster.broker.setVmMapper(this::map);
    }

    private Vm map(Cloudlet cloudlet) {
        VmState best = roundRobin ? nextRoundRobin() : leastOutstanding();
        if (best == null) {
            dropped.add(cloudlet.getId());
            return Vm.NULL;
        }
        best.outstanding++;
        best.cloudletsMapped++;
        if (mappingObserver != null) {
            mappingObserver.accept(new double[]{cloudlet.getId(), cluster.sim.clock(), best.id()});
        }
        return best.vm;
    }

    private VmState leastOutstanding() {
        VmState best = null;
        for (VmState s : cluster.all()) {          // ascending id, so ties go to the lowest id
            if (s.isEligible() && (best == null || s.outstanding < best.outstanding)) {
                best = s;
            }
        }
        return best;
    }

    private VmState nextRoundRobin() {
        VmState first = null;
        for (VmState s : cluster.all()) {
            if (!s.isEligible()) continue;
            if (first == null) first = s;
            if (s.id() > lastRoundRobinId) {
                lastRoundRobinId = s.id();
                return s;
            }
        }
        if (first != null) lastRoundRobinId = first.id();
        return first;
    }

    public int droppedCount() {
        return dropped.size();
    }
}
