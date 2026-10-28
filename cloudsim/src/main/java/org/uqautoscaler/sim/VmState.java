package org.uqautoscaler.sim;

import org.cloudsimplus.vms.Vm;

/** Bookkeeping the autoscaler keeps for each VM it created. */
public final class VmState {
    public final Vm vm;
    public final double requestedAt;
    public int outstanding;          // mapped and not yet finished requests
    public boolean draining;         // receives no new requests
    public boolean destroyRequested;
    /** True only while ScalingActuator asks the broker to destroy this VM (see Cluster's destruction function). */
    boolean destroyNow;
    public boolean destroyed;
    public double destroyedAt = Double.NaN;
    // per-interval accumulators (seconds)
    public double aliveS, servingS, busyS;
    public long cloudletsMapped;

    VmState(Vm vm, double requestedAt) {
        this.vm = vm;
        this.requestedAt = requestedAt;
    }

    public long id() {
        return vm.getId();
    }

    public boolean isAlive() {
        return !destroyed;
    }

    /** Created and finished starting up: can execute requests (draining VMs still finish theirs). */
    public boolean isServing() {
        return !destroyed && vm.isCreated() && !vm.isStartingUp();
    }

    /** Booting VMs are billed and draw idle power but serve nothing. */
    public boolean isBooting() {
        return !destroyed && !draining && (!vm.isCreated() || vm.isStartingUp());
    }

    public boolean isRunning() {
        return isServing() && !draining;
    }

    /** May receive new requests from the balancer. */
    public boolean isEligible() {
        return isRunning() && !destroyRequested;
    }

    public boolean isBusy() {
        return !vm.getCloudletScheduler().getCloudletExecList().isEmpty();
    }

    void resetInterval() {
        aliveS = servingS = busyS = 0;
    }
}
