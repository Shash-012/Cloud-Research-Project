package com.mycloudsim;

import java.util.Calendar;
import org.cloudbus.cloudsim.core.CloudSim;

public class App {
    public static void main(String[] args) {
        System.out.println("Starting CloudSim simulation...");

        int numUsers = 1;
        Calendar calendar = Calendar.getInstance();
        boolean traceFlag = false;

        CloudSim.init(numUsers, calendar, traceFlag);

        System.out.println("CloudSim initialized successfully!");
    }
}