package com.luoxq.ann;

import com.luoxq.ann.app.Mnist;
import org.junit.Ignore;
import org.junit.Test;

import static com.luoxq.ann.Math.maxIndex;
import static org.junit.Assert.assertTrue;

public class MnistTest {


    @Test
    public void test3Layers() {
        int epochs = 1;
        double rate = 1;
        int[] shape = {28 * 28, 50, 10};
        NeuralNetwork nn = new SigmoidNeuralNetwork(shape);
        nn.setLearningRate(rate);
        int correct = train(epochs, nn);
        assertTrue("correct rate", correct > 8000);
    }

    @Test
    public void test4Layers() {
        int epochs = 1;
        double rate = 1;
        int[] shape = {28 * 28, 20, 20, 10};
        NeuralNetwork nn = new SigmoidNeuralNetwork(shape);
        nn.setLearningRate(rate);
        int correct = train(epochs, nn);
        assertTrue("correct rate ", correct > 8000);
    }

    @Ignore
    @Test
    public void test5Layers() {
        int epochs = 1;
        double rate = 1;
        int[] shape = {28 * 28, 50, 50, 50, 10};
        NeuralNetwork nn = new SigmoidNeuralNetwork(shape);
        nn.setLearningRate(rate);
        int correct = train(epochs, nn);
        assertTrue("correct rate ", correct > 8000);
    }


    private int train(int epochs, NeuralNetwork nn) {
        Mnist mnist = new Mnist();
        mnist.load();
        mnist.shuffle();
        System.out.println("Network: " + nn.toJson());
        System.out.println("Initial correct rate: " + train(nn, mnist));
        System.out.println("Learning rate: " + nn.getLearningRate());
        System.out.println("Epoch,Time,Correctness\n----------------------");
        long time = System.currentTimeMillis();
        DataRecord[] data = mnist.getTrainingSlice(0, 60000);
        int correct = 0;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            Math.shuffle(data);
            for (int sample = 0; sample < data.length; sample++) {
                DataRecord row = data[sample];
                if (!row.correct)
                    nn.train(row.input, row.output);
            }
            long seconds = (System.currentTimeMillis() - time) / 1000;
            int result = train(nn, mnist);
            correct = java.lang.Math.max(correct, result);
            System.out.println(epoch + ", " + seconds + ", " + result);
        }
        return correct;
    }

    private static int train(NeuralNetwork nn, Mnist mnist) {
        int correct = 0;
        DataRecord[] data = mnist.getTestSlice(0, 10000);
        for (int sample = 0; sample < data.length; sample++) {
            DataRecord row = data[sample];
            if (maxIndex(nn.call(row.input)) == row.maxIndex) {
                correct++;
                row.correct = true;
            } else {
                row.correct = false;
            }
        }
        return correct;
    }
}
