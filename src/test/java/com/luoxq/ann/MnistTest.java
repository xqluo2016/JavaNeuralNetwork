package com.luoxq.ann;

import org.junit.Test;

import static com.luoxq.ann.Util.maxIndex;
import static org.junit.Assert.assertTrue;

public class MnistTest {


    @Test
    public void testSigmoidNetwork() {
        int epochs = 100;
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
        DataRecord[] data = mnist.getTrainingSlice(0, 60000);
        int correct = 0;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            Util.shuffle(data);
            for (int sample = 0; sample < data.length; sample++) {
                DataRecord row = data[sample];
                if (!row.correct)
                    nn.train(row.input, row.output);
            }
            long seconds = (System.currentTimeMillis() - time) / 1000;
            System.out.println(epoch + ", " + seconds + ", " +
                    (correct = test(nn, mnist)));
        }
        return correct;
    }

    private static int test(NeuralNetwork nn, Mnist mnist) {
        int correct = 0;
        DataRecord[] data = mnist.getTestSlice(0, 10000);
        for (int sample = 0; sample < data.length; sample++) {
            DataRecord row = data[sample];
            if (maxIndex(nn.f(row.input)) == row.maxIndex) {
                correct++;
                row.correct = true;
            } else {
                row.correct = false;
            }
        }
        return correct;
    }
}
