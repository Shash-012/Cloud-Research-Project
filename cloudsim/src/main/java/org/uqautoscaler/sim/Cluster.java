package org.uqautoscaler.sim;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.power.models.PowerModelHostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletScheduler;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerSpaceShared;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/** Engine, one datacenter with one single-PE host per possible VM, a broker and the VM registry (Step 3.3). */
public final class Cluster {
    public final SimConfig cfg;
    public final CloudSimPlus sim;
    public final DatacenterSimple datacenter;
    public final DatacenterBrokerSimple broker;
    public final List<Host> hosts = new ArrayList<>();
    /** All VMs ever created, by id (sorted, so iteration order is deterministic). */
    private final Map<Long, VmState> vms = new TreeMap<>();
    private long nextVmId = 0;

    public Cluster(SimConfig cfg, boolean cloudsimPowerModel) {
        this.cfg = cfg;
        this.sim = new CloudSimPlus(cfg.minTimeBetweenEvents);
        for (int h = 0; h < cfg.hosts; h++) {
            List<Pe> pes = new ArrayList<>();
            for (int p = 0; p < cfg.hostPes; p++) {
                pes.add(new PeSimple(cfg.hostMips * cfg.mipsScale));
            }
            HostSimple host = new HostSimple(1_000_000, 1_000_000, 10_000_000, pes);
            if (cloudsimPowerModel) {
                // linear model with the same idle and peak values, for the energy cross-check (Step 3.8)
                host.setPowerModel(new PowerModelHostSimple(cfg.pMaxW, cfg.pIdleW));
            }
            hosts.add(host);
        }
        this.datacenter = new AccurateDatacenter(sim, hosts);
        this.broker = new DatacenterBrokerSimple(sim);
        broker.setShutdownWhenIdle(false);
        // Automatic destruction of idle VMs stays disabled: the function returns the broker's default (never
        // destroy), which is also what setVmDestructionDelayFunction(null) installs in 8.5.7. The broker calls
        // requestIdleVmDestruction after every cloudlet return, so the function returns 0 ("destroy if idle")
        // only during ScalingActuator's own request for a drained VM with no outstanding requests.
        broker.setVmDestructionDelayFunction(vm -> {
            VmState st = vms.get(vm.getId());
            return st != null && st.destroyNow ? 0.0 : DatacenterBroker.DEF_VM_DESTRUCTION_DELAY;
        });
        for (int i = 0; i < cfg.initialVms; i++) {
            submitNewVm(0.0, 0.0);
        }
    }

    /** Create and submit one VM; it can serve only after `startupDelay` seconds. */
    public VmState submitNewVm(double now, double startupDelay) {
        Vm vm = new VmSimple(nextVmId++, cfg.engineMips(), cfg.vmPes);
        vm.setRam(1024).setBw(1000).setSize(10_000);
        vm.setCloudletScheduler(new CloudletSchedulerSpaceShared());
        vm.setStartupDelay(startupDelay);
        VmState st = new VmState(vm, now);
        vm.addOnHostDeallocationListener(info -> {
            st.destroyed = true;
            st.destroyedAt = info.getTime();
        });
        vms.put(vm.getId(), st);
        broker.submitVm(vm);
        return st;
    }

    public VmState state(Vm vm) {
        return vms.get(vm.getId());
    }

    public Collection<VmState> all() {
        return vms.values();
    }

    public int count(java.util.function.Predicate<VmState> p) {
        int n = 0;
        for (VmState s : vms.values()) {
            if (p.test(s)) n++;
        }
        return n;
    }

    public int nRunning() { return count(VmState::isRunning); }
    public int nBooting() { return count(VmState::isBooting); }
    public int nDraining() { return count(s -> s.draining && !s.destroyed); }
    public int nAlive() { return count(VmState::isAlive); }

    /**
     * The broker and the cloudlet schedulers keep every finished cloudlet in lists that are only appended to.
     * With millions of requests per run this exhausts memory, so finished entries are dropped at every
     * interval boundary. None of these lists is read by the simulation logic (checked in the 8.5.7 sources).
     */
    @SuppressWarnings("unchecked")
    public void releaseFinishedCloudlets() {
        broker.getCloudletSubmittedList().removeIf(Cloudlet::isFinished);
        broker.getCloudletCreatedList().removeIf(Cloudlet::isFinished);
        clearField(broker, "cloudletFinishedList");
        for (VmState s : vms.values()) {
            CloudletScheduler sch = s.vm.getCloudletScheduler();
            sch.getCloudletFinishedList().clear();
            clearField(sch, "cloudletReturnedList");
        }
    }

    private static void clearField(Object target, String name) {
        Class<?> c = target.getClass();
        while (c != null) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                ((Collection<?>) f.get(target)).clear();
                return;
            } catch (NoSuchFieldException e) {
                c = c.getSuperclass();
            } catch (IllegalAccessException e) {
                throw new IllegalStateException(e);
            }
        }
        throw new IllegalStateException("field " + name + " not found on " + target.getClass());
    }
}
