package org.uqautoscaler.sim;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimEntity;
import org.cloudsimplus.core.events.SimEvent;
import org.cloudsimplus.utilizationmodels.UtilizationModel;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Turns the per-interval rate series into Poisson arrivals (Step 3.4).
 *
 * <p>In CloudSim Plus 8.5.7 the broker calls the VM mapper when it processes the creation request that follows
 * {@code submitCloudletList}, i.e. at submission time, not at the cloudlet's arrival. Submitting a whole interval
 * up front with submission delays would therefore map every request up to a minute early. Instead this entity
 * schedules one ARRIVAL event per request and submits that request (delay 0) when the event fires, so mapping
 * happens exactly at the planned arrival time (verified by ArrivalTimingTest).</p>
 */
public final class ArrivalGenerator extends CloudSimEntity {
    static final int ARRIVAL = 900_001;
    static final int BOUNDARY = 900_002;

    private final Cluster cluster;
    private final double[] rates;            // unscaled requests/s per interval (as exported by Python)
    private final SplittableRandom arrivalRng;
    private final SplittableRandom lengthRng;
    private final UtilizationModel full = new UtilizationModelFull();
    private final Map<Long, Double> plannedArrival = new HashMap<>();
    private long nextCloudletId = 0;
    private final Telemetry telemetry;

    public ArrivalGenerator(Cluster cluster, double[] rates, long seed, Telemetry telemetry) {
        super(cluster.sim);
        this.cluster = cluster;
        this.rates = rates;
        this.arrivalRng = new SplittableRandom(seed * 1_000_003L + 11);
        this.lengthRng = new SplittableRandom(seed * 1_000_003L + 29);
        this.telemetry = telemetry;
    }

    @Override
    protected void startInternal() {
        // the first interval's arrivals are scheduled when the simulation starts
        scheduleInterval(0);
    }

    /** Schedule the Poisson arrivals of interval t, which starts now. */
    public void scheduleInterval(int t) {
        double interval = cluster.cfg.intervalS;
        schedule(this, interval, BOUNDARY);      // guarantees a clock tick at the interval's end
        if (t >= rates.length) return;
        double lambda = rates[t] * cluster.cfg.capacityScale;
        if (lambda <= 0) return;
        double meanLen = cluster.cfg.meanLengthMi();
        double offset = 0;
        while (true) {
            offset += -Math.log(1.0 - arrivalRng.nextDouble()) / lambda;
            if (offset >= interval) break;
            long length = Math.max(1L, Math.round(-Math.log(1.0 - lengthRng.nextDouble()) * meanLen));
            schedule(this, offset, ARRIVAL, length);
        }
    }

    @Override
    public void processEvent(SimEvent evt) {
        if (evt.getTag() != ARRIVAL) return;
        long length = (Long) evt.getData();
        Cloudlet c = new CloudletSimple(nextCloudletId, length, 1);
        c.setUtilizationModelCpu(full);
        c.setUtilizationModelRam(UtilizationModel.NULL);
        c.setUtilizationModelBw(UtilizationModel.NULL);
        c.setSizes(1);
        double now = getSimulation().clock();
        plannedArrival.put(nextCloudletId, now);
        nextCloudletId++;
        c.addOnFinishListener(info -> onFinish(info.getCloudlet()));
        telemetry.onArrival();
        cluster.broker.submitCloudletList(List.of(c));
    }

    private void onFinish(Cloudlet c) {
        Double arrival = plannedArrival.remove(c.getId());
        VmState st = cluster.state(c.getVm());
        if (st != null) {
            st.outstanding--;
        }
        telemetry.onCompletion((c.getFinishTime() - arrival) * 1000.0);
        if (st != null) {
            ScalingActuator.destroyIfDrained(cluster, st);
        }
    }

    public double plannedArrivalOf(long cloudletId) {
        return plannedArrival.getOrDefault(cloudletId, Double.NaN);
    }

    public long generated() {
        return nextCloudletId;
    }
}
