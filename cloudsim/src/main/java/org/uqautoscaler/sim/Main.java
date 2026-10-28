package org.uqautoscaler.sim;

import org.cloudsimplus.core.CloudSimPlus;

public class Main {
    public static void main(String[] args) {
        CloudSimPlus sim = new CloudSimPlus();
        System.out.println("CloudSim Plus engine created; clock = " + sim.clock());
    }
}
