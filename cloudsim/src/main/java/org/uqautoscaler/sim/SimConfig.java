package org.uqautoscaler.sim;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Scenario settings. Python passes a JSON copy of configs/cloudsim.yaml (Step 3.2, "SimConfig");
 * {@link #defaults()} mirrors that file for tests.
 */
public final class SimConfig {
    public double intervalS = 60;
    public int stepsPerRun = 360;
    public int hosts = 30;
    public int hostPes = 1;
    public long hostMips = 1000;
    public int vmPes = 1;
    public long vmMips = 1000;
    public int vmMin = 1;
    public int vmMax = 30;
    public int initialVms = 4;
    public double vmStartupDelayS = 30;
    public double meanServiceS = 0.04;
    public double requestKb = 8;
    public double capacityScale = 1.0;
    public String balancer = "least_outstanding";
    public double sloMs = 250.0;              // unscaled; effective SLO is sloMs / capacityScale
    public int kMin = -7;
    public int kMax = 7;
    public double pricePerVmHour = 0.10;
    public String powerModel = "square";
    public double pIdleW = 80.0;
    public double pMaxW = 400.0;
    public double minTimeBetweenEvents = 0.0001;
    /** Instructions are counted in units of 1/mipsScale MI; CloudSim Plus truncates processed work to whole units. */
    public long mipsScale = 1000;
    public double memOffset = 30.0;
    public double memSlope = 45.0;
    public double memNoiseStd = 2.0;
    public double rtUMin = 0.3;
    public double rtUMax = 1.1;
    public int rtHoldIntervals = 2;
    public double cpuUpper = 80, cpuLower = 30;
    public int cpuCooldown = 3;
    public double rtUpperFrac = 0.8, rtLowerFrac = 0.4;
    public int rtCooldown = 3;
    public String bridgeHost = "127.0.0.1";
    public int bridgePort = 5555;
    public double bridgeTimeoutS = 120;

    /** Raw JSON as received, echoed to Python in the hello message. */
    public JsonNode raw;

    public static SimConfig defaults() {
        return new SimConfig();
    }

    public double effectiveSloMs() {
        return sloMs / capacityScale;
    }

    /** Mean request length in MI: mean_service_s * vm_mips / s (Step 3.1, "Run time"). */
    public double meanLengthMi() {
        return meanServiceS * vmMips * mipsScale / capacityScale;
    }

    /** VM (and host PE) capacity in the engine's instruction units. */
    public long engineMips() {
        return vmMips * mipsScale;
    }

    /** Service rate of one VM in requests per simulated second. */
    public double muPerVm() {
        return (double) engineMips() / meanLengthMi();
    }

    public static SimConfig fromJson(Path path) throws IOException {
        JsonNode n = new ObjectMapper().readTree(path.toFile());
        SimConfig c = new SimConfig();
        c.raw = n;
        c.intervalS = d(n, "time/interval_s", c.intervalS);
        c.stepsPerRun = i(n, "time/steps_per_run", c.stepsPerRun);
        c.hosts = i(n, "cluster/hosts", c.hosts);
        c.hostPes = i(n, "cluster/host_pes", c.hostPes);
        c.hostMips = (long) d(n, "cluster/host_mips", c.hostMips);
        c.vmPes = i(n, "cluster/vm_pes", c.vmPes);
        c.vmMips = (long) d(n, "cluster/vm_mips", c.vmMips);
        c.vmMin = i(n, "cluster/vm_min", c.vmMin);
        c.vmMax = i(n, "cluster/vm_max", c.vmMax);
        c.initialVms = i(n, "cluster/initial_vms", c.initialVms);
        c.vmStartupDelayS = d(n, "cluster/vm_startup_delay_s", c.vmStartupDelayS);
        c.meanServiceS = d(n, "requests/mean_service_s", c.meanServiceS);
        c.requestKb = d(n, "requests/request_kb", c.requestKb);
        c.capacityScale = d(n, "requests/capacity_scale", c.capacityScale);
        c.balancer = s(n, "balancer", c.balancer);
        c.sloMs = d(n, "slo/rt_ms", c.sloMs);
        c.kMin = i(n, "actions/k_min", c.kMin);
        c.kMax = i(n, "actions/k_max", c.kMax);
        c.pricePerVmHour = d(n, "cost/price_per_vm_hour", c.pricePerVmHour);
        c.powerModel = s(n, "power/model", c.powerModel);
        c.pIdleW = d(n, "power/p_idle_w", c.pIdleW);
        c.pMaxW = d(n, "power/p_max_w", c.pMaxW);
        c.minTimeBetweenEvents = d(n, "engine/min_time_between_events", c.minTimeBetweenEvents);
        c.mipsScale = (long) d(n, "engine/mips_scale", c.mipsScale);
        c.memOffset = d(n, "telemetry/mem_util/offset", c.memOffset);
        c.memSlope = d(n, "telemetry/mem_util/slope", c.memSlope);
        c.memNoiseStd = d(n, "telemetry/mem_util/noise_std", c.memNoiseStd);
        c.rtUMin = d(n, "random_target/u_min", c.rtUMin);
        c.rtUMax = d(n, "random_target/u_max", c.rtUMax);
        c.rtHoldIntervals = i(n, "random_target/hold_intervals", c.rtHoldIntervals);
        c.cpuUpper = d(n, "baselines/cpu_threshold/upper", c.cpuUpper);
        c.cpuLower = d(n, "baselines/cpu_threshold/lower", c.cpuLower);
        c.cpuCooldown = i(n, "baselines/cpu_threshold/cooldown_steps", c.cpuCooldown);
        c.rtUpperFrac = d(n, "baselines/rt_threshold/upper_frac_slo", c.rtUpperFrac);
        c.rtLowerFrac = d(n, "baselines/rt_threshold/lower_frac_slo", c.rtLowerFrac);
        c.rtCooldown = i(n, "baselines/rt_threshold/cooldown_steps", c.rtCooldown);
        c.bridgeHost = s(n, "bridge/host", c.bridgeHost);
        c.bridgePort = i(n, "bridge/port", c.bridgePort);
        c.bridgeTimeoutS = d(n, "bridge/timeout_s", c.bridgeTimeoutS);
        return c;
    }

    private static JsonNode at(JsonNode n, String path) {
        return n.at("/" + path);
    }

    private static double d(JsonNode n, String p, double def) {
        JsonNode v = at(n, p);
        return v.isMissingNode() || v.isNull() ? def : v.asDouble();
    }

    private static int i(JsonNode n, String p, int def) {
        JsonNode v = at(n, p);
        return v.isMissingNode() || v.isNull() ? def : v.asInt();
    }

    private static String s(JsonNode n, String p, String def) {
        JsonNode v = at(n, p);
        return v.isMissingNode() || v.isNull() ? def : v.asText();
    }
}
