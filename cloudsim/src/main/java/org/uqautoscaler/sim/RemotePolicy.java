package org.uqautoscaler.sim;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * Asks the Python bridge for every decision (Step 3.11): newline-delimited JSON over one TCP connection.
 * If Python does not answer within bridge.timeout_s the run aborts with a clear error instead of hanging.
 */
public final class RemotePolicy implements Policy {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final Socket socket;
    private final BufferedReader in;
    private final Writer out;
    private boolean fallback;

    public RemotePolicy(SimConfig cfg, String host, int port, String runId, String policy, long seed) {
        try {
            socket = new Socket();
            int timeoutMs = (int) (cfg.bridgeTimeoutS * 1000);
            socket.connect(new InetSocketAddress(host, port), timeoutMs);
            socket.setSoTimeout(timeoutMs);
            socket.setTcpNoDelay(true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
            out = new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8);
            ObjectNode hello = JSON.createObjectNode();
            hello.put("type", "hello");
            hello.put("run_id", runId);
            hello.put("policy", policy);
            hello.put("seed", seed);
            hello.set("config", cfg.raw == null ? JSON.createObjectNode() : cfg.raw);
            JsonNode reply = exchange(hello);
            if (!"ready".equals(reply.path("type").asText())) {
                throw new IllegalStateException("bridge: expected ready, got " + reply);
            }
        } catch (IOException e) {
            throw new IllegalStateException("bridge: cannot connect to " + host + ":" + port + ": " + e.getMessage(), e);
        }
    }

    private JsonNode exchange(ObjectNode msg) throws IOException {
        out.write(JSON.writeValueAsString(msg));
        out.write('\n');
        out.flush();
        String line;
        try {
            line = in.readLine();
        } catch (SocketTimeoutException e) {
            throw new IllegalStateException("bridge: no answer from Python within the timeout", e);
        }
        if (line == null) {
            throw new IllegalStateException("bridge: Python closed the connection");
        }
        JsonNode reply = JSON.readTree(line);
        if ("error".equals(reply.path("type").asText())) {
            throw new IllegalStateException("bridge: Python reported an error: " + reply.path("message").asText());
        }
        return reply;
    }

    private static void putNum(ObjectNode n, String key, double v) {
        if (Double.isFinite(v)) n.put(key, v); else n.putNull(key);
    }

    @Override
    public int decide(Observation obs) {
        ObjectNode m = JSON.createObjectNode();
        m.put("type", "observe");
        m.put("step", obs.step);
        m.put("time_s", obs.timeS);
        m.put("n_running", obs.nRunning);
        m.put("n_booting", obs.nBooting);
        m.put("n_draining", obs.nDraining);
        ObjectNode metrics = m.putObject("metrics");
        obs.metrics.forEach((k, v) -> putNum(metrics, k, v));
        putNum(m, "rt_mean_ms", obs.rtMeanMs);
        putNum(m, "rate", obs.rate);
        try {
            JsonNode reply = exchange(m);
            if (!"action".equals(reply.path("type").asText())) {
                throw new IllegalStateException("bridge: expected action, got " + reply);
            }
            fallback = reply.path("fallback").asBoolean(false);
            return reply.path("k").asInt();
        } catch (IOException e) {
            throw new IllegalStateException("bridge: " + e.getMessage(), e);
        }
    }

    @Override
    public boolean lastFallback() {
        return fallback;
    }

    @Override
    public void close(Map<String, Object> summary) {
        try {
            ObjectNode bye = JSON.createObjectNode();
            bye.put("type", "bye");
            bye.set("summary", JSON.valueToTree(summary));
            out.write(JSON.writeValueAsString(bye));
            out.write('\n');
            out.flush();
            socket.close();
        } catch (IOException ignored) {
            // the run itself is complete; a failed goodbye does not invalidate it
        }
    }
}
