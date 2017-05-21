package com.luoxq.ann;

import org.junit.Test;

import java.util.Arrays;

import static com.luoxq.ann.Util.maxIndex;
import static org.junit.Assert.assertTrue;

public class MnistTest {


    @Test
    public void testSigmoidNetwork() {
        int epochs = 2;
        double rate = 0.5;
        int[] shape = {28 * 28, 50, 10};
        NeuralNetwork nn = new SigmoidNeuralNetwork(shape);
        nn.setLearningRate(rate);
        int correct = test(epochs, nn);
        assertTrue("correct rate >5000", correct > 9000);
    }

    private int test(int epochs, NeuralNetwork nn) {
        Mnist mnist = new Mnist();
        mnist.load();
        mnist.shuffle();
        System.out.println("Network: " + nn.toJson());
        System.out.println("Initial correct rate: " + test(nn, mnist));
        System.out.println("Learning rate: " + nn.getLearningRate());
        System.out.println("Epoch,Time,Correctness\n----------------------");
        long time = System.currentTimeMillis();
        Mnist.Data[] data = mnist.getTrainingSlice(0, 60000);
        int correct = 0;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int sample = 0; sample < data.length; sample++) {
                nn.train(data[sample].input, data[sample].output);
            }
            long seconds = (System.currentTimeMillis() - time) / 1000;
            System.out.println(epoch + ", " + seconds + ", " +
                    (correct = test(nn, mnist)));
        }
        return correct;
    }

    private static int test(NeuralNetwork nn, Mnist mnist) {
        int correct = 0;
        Mnist.Data[] data = mnist.getTestSlice(0, 10000);
        for (int sample = 0; sample < data.length; sample++) {
            if (maxIndex(nn.f(data[sample].input)) == data[sample].label) {
                correct++;
            }
        }
        return correct;
    }
}
