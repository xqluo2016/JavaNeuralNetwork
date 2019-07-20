package com.luoxq.ann;

import org.junit.Test;

public class SingleLinearNeuronTest {



    @Test
    public void test() {
        SingleLinearNeuron n = new SingleLinearNeuron(0, 0);
        //target: y = 3*x + 3;
        double rate = 0.1;
        int epoch = 100;
        int trainingSize = 200;
        for (int i = 0; i < epoch; i++) {
            double[][] data = n.generateTrainingData(trainingSize);
            n.train(data, rate);
            System.out.printf("Epoch: %3d,  W: %f, B: %f \n", i, n.w, n.b);
        }
        assert(Math.abs(n.w -3)<0.1);
        assert(Math.abs(n.b -3)<0.1);
    }
}
