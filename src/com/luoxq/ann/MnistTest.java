package com.luoxq.ann;

import java.util.Arrays;

public class MnistTest {


    public static void main(String... args) {
        int[] shape = {28 * 28, 40, 10, 10};
        SigmoidNetwork nn = new SigmoidNetwork(shape);
        Mnist mnist = new Mnist();
        mnist.load();
        mnist.shuffle();
        System.out.println("Shape: " + Arrays.toString(shape));
        System.out.println("Initial correct rate: " + test(nn, mnist));
        int epochs = 1000;
        double rate = 0.5;
        System.out.println("Learning rate: " + rate);
        System.out.println("Epoch,Time,Correctness\n----------------------");
        long time = System.currentTimeMillis();
        Mnist.Data[] data = mnist.getTrainingSlice(0, 60000);

        nn.setLearningRate(rate);
        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int sample = 0; sample < data.length; sample++) {
                nn.train(data[sample].input, data[sample].output);
            }
            long seconds = (System.currentTimeMillis() - time) / 1000;
            System.out.println(epoch + ", " + seconds + ", " +
                    test(nn, mnist));
        }
    }

    private static int test(SigmoidNetwork nn, Mnist mnist) {
        int correct = 0;
        Mnist.Data[] data = mnist.getTestSlice(0, 10000);
        for (int sample = 0; sample < data.length; sample++) {
            if (max(nn.f(data[sample].input)) == data[sample].label) {
                correct++;
            }
        }
        return correct;
    }

    private static int max(double[] d) {
        double max = d[0];
        int idx = 0;
        for (int i = 1; i < d.length; i++) {
            if (max < d[i]) {
                max = d[i];
                idx = i;
            }
        }
        return idx;
    }
}
