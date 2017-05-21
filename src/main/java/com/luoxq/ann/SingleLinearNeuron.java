package com.luoxq.ann;

import java.util.Random;

public class SingleLinearNeuron {
    double weight;
    double bias;

    public SingleLinearNeuron(double weight, double bias) {
        this.weight = weight;
        this.bias = bias;
    }

    public double f(double x) {
        return x * weight + bias;
    }

    public double cost(double x, double y) {
        return f(x) - y;
    }

    // c = w*x + b - y
    public double[] gradient(double x, double y) {
        double c = cost(x, y);
        double dw = x * c;
        double db = 1 * c;
        double d = Math.max(Math.abs(dw), Math.abs(db));
        if (d == 0) {
            d = 1;
        }
        return new double[]{-dw / d, -db / d};
    }

    public void train(double[][] data, double rate) {
        for (int i = 0; i < data.length; i++) {
            double x = data[i][0];
            double y = data[i][1];
            double[] gradient = gradient(x, y);
            weight += gradient[0] * rate;
            bias += gradient[1] * rate;
        }
    }

    protected SingleLinearNeuron getTarget() {
        return new SingleLinearNeuron(3, 3);
    }

    public double[][] generateTrainingData(int size) {
        Random rand = new Random(System.nanoTime());
        double[][] data = new double[size][];
        SingleLinearNeuron target = getTarget();
        for (int i = 0; i < data.length; i++) {
            double x = rand.nextDouble() * 100;
            double y = target.f(x);
            data[i] = new double[]{x, y};
        }
        return data;
    }

    public static void main(String... args) {
        SingleLinearNeuron n = new SingleLinearNeuron(0, 0);
        //target: y = 3*x + 3;
        double rate = 0.1;
        int epoch = 100;
        int trainingSize = 20;
        for (int i = 0; i < epoch; i++) {
            double[][] data = n.generateTrainingData(trainingSize);
            n.train(data, rate);
            System.out.printf("Epoch: %3d,  W: %f, B: %f \n", i, n.weight, n.bias);
        }
    }
}
