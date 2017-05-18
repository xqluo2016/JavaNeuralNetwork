package com.luoxq.ann;

import java.text.NumberFormat;

public class Test7 {

    static double[][] x = {
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

    static double[][] expect = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
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

    public static void main(String... args) {
        SigmoidNetwork nn = new SigmoidNetwork(7, 10);


        System.out.println("Testing: ");
        double cost = getCost(nn);
        System.out.println("Cost before train: " + cost);
        System.out.println("Dump: \n" + nn.toJson().replace("],", "],\n"));
        System.out.println("Training...");
        int epochs = 1000;
        double rate = 10;
        nn.setLearningRate(rate);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int sample = 0; sample < x.length; sample++)
                nn.train(x[sample], expect[sample]);
        }
        System.out.println("Testing: ");
        cost = getCost(nn);
        System.out.println("Cost after train: " + cost);
        System.out.println("Dump: \n" + nn.toJson().replace("],", "],\n"));
    }


    static double getCost(SigmoidNetwork nn) {
        double cost = 0;
        for (int i = 0; i < x.length; i++) {
            double[] y = nn.f(x[i]);
            System.out.println("output of " + i + ": " + toString(y));
            double[] exp = expect[i];
            cost += getCost(y, exp);
        }
        return cost;
    }

    static double getCost(double[] y, double[] exp) {
        double cost = 0;
        for (int j = 0; j < y.length; j++) {
            double diff = Math.abs(y[j] - exp[j]);
            cost += diff;
        }
        return cost;
    }

    static String toString(double[] d) {
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
