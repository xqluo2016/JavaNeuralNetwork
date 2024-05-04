package com.luoxq.ann;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertTrue;

public class SigmoidNeuralNetworkTest {

    @Test
    public void testResultShouldNear05() {
        double total = 0;
        int count = 100;
        double[] in = new double[900];
        Arrays.fill(in, 1);
        for (int i = 0; i < count; i++) {
            SigmoidNeuralNetwork n = new SigmoidNeuralNetwork(900, 30, 30, 1);
            double v = n.f(in)[0];
            total += v;
            //System.out.println(v);
        }
        total /= count;
        System.out.println("AVG: " + total);
        assertTrue(Math.abs(total - 0.5) < 0.1);
    }
}
