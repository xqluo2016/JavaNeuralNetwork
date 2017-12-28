package com.luoxq.ann;

import org.junit.Test;

import java.text.NumberFormat;

import static org.junit.Assert.assertTrue;

public class SevenDigitTest {

    private static final double[][] x = {
            {1, 1, 1, 0, 1, 1, 1},
            {0, 0, 1, 0, 0, 1, 0},
            {1, 0, 1, 1, 1, 0, 1},
            {1, 0, 1, 1, 0, 1, 1},
            {0, 1, 1, 1, 0, 1, 0},
            {1, 1, 0, 1, 0, 1, 1},
            {1, 1, 0, 1, 1, 1, 1},
            {1, 0, 1, 0, 0, 1, 0},
            {1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 0, 1, 1}
    };

    private static final double[][] expect = {
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    };

    @Test
    public void testSigmoidNeuralNetwork() {
        NeuralNetwork nn = new SigmoidNeuralNetwork(7, 10, 10);
        test(nn);
    }

    @Test
    public void testLinkedNeuralNetwork() {
        SigmoidNeuralNetwork s1 = new SigmoidNeuralNetwork(7, 10);
        SigmoidNeuralNetwork s2 = new SigmoidNeuralNetwork(10, 10);

        NeuralNetwork nn = new LinkedNeuralNetwork(s1, s2);
        test(nn);
    }

    @Test
    public void testParallelNeuralNetwork() {
        SigmoidNeuralNetwork s1 = new SigmoidNeuralNetwork(5, 5);
        SigmoidNeuralNetwork s2 = new SigmoidNeuralNetwork(5, 5);

        NeuralNetwork p = new ParallelNeuralNetwork(s1, s2);
        NeuralNetwork nn = new LinkedNeuralNetwork(new SigmoidNeuralNetwork(7, 10), p);
        test(nn);
    }

    private void test(NeuralNetwork nn) {
        System.out.println("Testing: ");
        double cost = check(nn);
        System.out.println("Cost before train: " + cost);
        System.out.println("Dump: \n" + nn.toJson().replace("],", "],\n"));
        System.out.println("Training...");
        int epochs = 1000;
        double rate = 5;
        nn.setLearningRate(rate);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int sample = 0; sample < x.length; sample++)
                nn.train(x[sample], expect[sample]);
        }
        System.out.println("Testing: ");
        cost = check(nn);
        System.out.println("Cost after train: " + cost);
        assertTrue("cost after train", cost < 1);
    }


    double check(NeuralNetwork nn) {
        double cost = 0;
        for (int i = 0; i < x.length; i++) {
            double[] y = nn.call(x[i]);
            System.out.println("output of " + i + ": " + toString(y));
            double[] exp = expect[i];
            cost += getCost(y, exp);
        }
        return cost;
    }

    double getCost(double[] y, double[] exp) {
        double cost = 0;
        for (int j = 0; j < y.length; j++) {
            double diff = Math.abs(y[j] - exp[j]);
            cost += diff;
        }
        return cost;
    }

    String toString(double[] d) {
        NumberFormat nf = NumberFormat.getInstance();
        nf.setMaximumFractionDigits(2);
        nf.setMinimumFractionDigits(2);
        StringBuilder sb = new StringBuilder();
        for (double dd : d) {
            sb.append(nf.format(dd)).append(",");
        }
        return sb.toString();
    }

}
