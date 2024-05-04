package com.luoxq.ann;

import org.junit.Test;

import java.util.Random;

public class SingleLinearNeuronTest {

    protected SingleLinearNeuron target = new SingleLinearNeuron(3, 3);

    public double[][] generateTrainingData(int size) {
        Random rand = new Random(System.nanoTime());
        double[][] data = new double[size][];
        for (int i = 0; i < data.length; i++) {
            double x = rand.nextDouble() * 100;
            double y = target.f(x);
            data[i] = new double[]{x, y};
        }
        return data;
    }


    @Test
    public void test() {
        SingleLinearNeuron n = new SingleLinearNeuron(0, 0);
        //target: y = 3*x + 3;
        double rate = 0.001;
        int epoch = 100;
        int trainingSize = 200;
        for (int i = 0; i < epoch; i++) {
            double[][] data = generateTrainingData(trainingSize);
            n.train(data, rate);
            System.out.printf("Epoch: %3d,  W: %f, B: %f \n", i, n.w, n.b);
        }
        assert (Math.abs(n.w - 3) < 0.1);
        assert (Math.abs(n.b - 3) < 0.1);
    }
}
