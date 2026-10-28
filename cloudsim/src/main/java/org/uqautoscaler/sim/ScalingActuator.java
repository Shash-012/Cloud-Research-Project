package org.uqautoscaler.sim;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Applies a scaling action k (Step 3.6). */
public final class ScalingActuator {
    private final Cluster cluster;

    public ScalingActuator(Cluster cluster) {
        this.cluster = cluster;
    }

    /**
     * Returns the action actually applied after clamping: running + booting stays within [vm_min, vm_max], and
     * running + booting + draining never exceeds the number of hosts (one VM per host).
     */
    public int apply(int k) {
        SimConfig cfg = cluster.cfg;
        double now = cluster.sim.clock();
        int w = cluster.nRunning() + cluster.nBooting();
        int target = Math.max(cfg.vmMin, Math.min(cfg.vmMax, w + k));
        int kEff = target - w;
        if (kEff > 0) {
            int free = cfg.hosts - cluster.nAlive();
            kEff = Math.min(kEff, Math.max(0, free));
            for (int i = 0; i < kEff; i++) {
                cluster.submitNewVm(now, cfg.vmStartupDelayS);
            }
        } else if (kEff < 0) {
            List<VmState> booting = new ArrayList<>();
            List<VmState> running = new ArrayList<>();
            for (VmState s : cluster.all()) {
                if (s.isBooting()) booting.add(s);
                else if (s.isRunning()) running.add(s);
            }
            // booting first (most recently requested first), then running with the fewest outstanding requests
            booting.sort(Comparator.comparingDouble((VmState s) -> -s.requestedAt).thenComparing(s -> -s.id()));
            running.sort(Comparator.comparingInt((VmState s) -> s.outstanding).thenComparing(s -> -s.id()));
            List<VmState> order = new ArrayList<>(booting);
            order.addAll(running);
            for (int i = 0; i < -kEff; i++) {
                VmState s = order.get(i);
                s.draining = true;
                destroyIfDrained(cluster, s);
            }
        }
        return kEff;
    }

    /** A draining VM with no outstanding requests is handed to the broker for destruction. */
    static void destroyIfDrained(Cluster cluster, VmState s) {
        if (s.draining && !s.destroyRequested && !s.destroyed && s.outstanding == 0 && s.vm.isCreated()) {
            s.destroyNow = true;
            cluster.broker.requestIdleVmDestruction(s.vm);
            s.destroyNow = false;
            // the broker shuts the VM down only if its execution list is already empty; otherwise retry later
            s.destroyRequested = s.vm.getShutdownBeginTime() >= 0;
        }
    }

    /** Retry drained VMs whose destruction could not start yet (called on every clock tick). */
    void retryPendingDestructions() {
        for (VmState s : cluster.all()) {
            destroyIfDrained(cluster, s);
        }
    }
}
