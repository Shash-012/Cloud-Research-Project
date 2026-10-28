package org.uqautoscaler.sim;

import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.core.Simulation;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;

import java.util.List;

/**
 * DatacenterSimple with exact processing-update timing. Two behaviours of 8.5.7 bias requests of tens of ms:
 * <ol>
 *   <li>updateHostsProcessing() never schedules the next update sooner than minTimeBetweenEvents + 0.01 s, so a
 *       request whose remaining work needs less than 11 ms finishes up to 11 ms late;</li>
 *   <li>updateCloudletProcessing() silently skips an update that comes less than minTimeBetweenEvents after the
 *       previous one, without rescheduling. With the 10 ms padding removed, the update scheduled for a request's
 *       finish can be skipped this way, and since arrivals to a busy VM trigger no update, the VM then stalls.</li>
 * </ol>
 * Here the next update is due at the earliest estimated finish (at least minTimeBetweenEvents ahead), every update
 * is processed except a repeat at the same instant, and a new update event is scheduled only if it is earlier than
 * the one already pending, so duplicate events cannot accumulate. Host and VM processing are inherited unchanged.
 */
public class AccurateDatacenter extends DatacenterSimple {
    private double nextUpdateAt = -1;
    private double lastUpdate = -1;

    public AccurateDatacenter(Simulation simulation, List<? extends Host> hostList) {
        super(simulation, hostList);
    }

    @Override
    protected double updateHostsProcessing() {
        double next = Double.MAX_VALUE;
        double clock = getSimulation().clock();
        for (Host host : getHostList()) {
            next = Math.min(next, host.updateProcessing(clock));
        }
        // a zero delay would re-trigger this update at the same instant forever (the base class relied on its
        // skip guard to break that loop), so every update is at least minTimeBetweenEvents ahead
        return Math.max(next, getSimulation().getMinTimeBetweenEvents());
    }

    @Override
    protected double updateCloudletProcessing() {
        double now = getSimulation().clock();
        // A second update at the same instant has no elapsed time to process. Skipping it also breaks a loop in
        // 8.5.7: with schedulingInterval 0, VmAbstract.bootProcessing() schedules the next update of a booting VM
        // with delay 0 (it tests schedulingInterval > -1), which the base class absorbed with its skip guard.
        if (now <= lastUpdate) {
            return Double.MAX_VALUE;
        }
        lastUpdate = now;
        double delay = updateHostsProcessing();
        if (delay != Double.MAX_VALUE) {
            double at = now + delay;
            if (nextUpdateAt <= now || at < nextUpdateAt) {
                schedule(delay, CloudSimTag.VM_UPDATE_CLOUDLET_PROCESSING);
                nextUpdateAt = at;
            }
        }
        setLastProcessTime(now);
        return delay;
    }
}
