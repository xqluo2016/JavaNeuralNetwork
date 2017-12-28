package com.luoxq.ann;

import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Random;

import static java.lang.Math.exp;
import static com.luoxq.ann.Util.*;

public class SigmoidNeuralNetwork implements NeuralNetwork {

    private static final long serialVersionUID = 1578550361939502383L;

    int[] shape;
    int layers;
    double learningRate = 1;
    double[][][] weights;
    double[][] bias;
    double[][] zs;
    double[][] xs;

    int inputSize;
    int outputSize;

    public SigmoidNeuralNetwork() {
    }

    public SigmoidNeuralNetwork(int... shape) {
        this.shape = shape;
        this.inputSize = shape[0];
        this.outputSize = shape[shape.length - 1];
        layers = shape.length;
        weights = new double[layers][][];
        bias = new double[layers][];
        //First layer is input layer, no weight
        weights[0] = new double[0][0];
        bias[0] = new double[0];
        zs = new double[layers][];
        xs = new double[layers][];
        for (int i = 1; i < layers; i++) {
            weights[i] = new double[this.shape[i]][this.shape[i - 1]];
            bias[i] = new double[this.shape[i]];
        }
        fillRandom(weights);
        fillRandom(bias);
    }

    public String toJson() {
        return "{type:'" + this.getClass().getName() + "', shape:" + Arrays.toString(shape) + ",weights:"
                + Arrays.deepToString(weights) + ",bias:" + Arrays.deepToString(bias) + "}";
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

    @Override
    public double[] f(double[] in) {
        zs[0] = xs[0] = in;
        for (int i = 1; i < layers; i++) {
            zs[i] = add(wx(xs[i - 1], weights[i]), bias[i]);
            xs[i] = sigmoid(zs[i]);
        }
        return xs[layers - 1];
    }


    double[] wx(double[] x, double[][] weight) {
        int numberOfNeron = weight.length;
        double[] wx = new double[numberOfNeron];
        for (int i = 0; i < numberOfNeron; i++) {
            wx[i] = dot(weight[i], x);//SUM(w*x)
        }
        return wx;
    }

    @Override
    public double[] train(double[] in, double[] expect) {
        double[] y = f(in);
        double[] cost = sub(expect, y);
        return train(cost);
    }

    public double[] train(double[] cost) {
        double[][][] dw = new double[layers][][];
        double[][] db = new double[layers][];
        dw[0] = new double[0][0];
        db[0] = new double[0];
        for (int i = layers - 1; i > 0; i--) {
            double[] sp = signmoidPrime(zs[i]);
            cost = mul(cost, sp);
            dw[i] = dw(xs[i - 1], cost);
            db[i] = cost;
            cost = dx(weights[i], cost);
        }
        weights = add(weights, mul(dw, learningRate));
        bias = add(bias, mul(db, learningRate));
        return cost;
    }

    @Override
    public int getInputSize() {
        return inputSize;
    }

    @Override
    public int getOutputSize() {
        return outputSize;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    //derivative of x is w*c and sum for each x
    double[] dx(double[][] w, double[] c) {
        int numberOfX = w[0].length;
        double[] v = new double[numberOfX];
        for (int i = 0; i < numberOfX; i++) {
            for (int j = 0; j < c.length; j++) {
                v[i] += w[j][i] * c[j];
            }
        }
        return v;
    }

    //derivative of w is x*c for each c and each x
    double[][] dw(double[] x, double[] c) {
        int numberOfNeuron = c.length;
        int numberOfIn = x.length;
        double[][] dw = new double[numberOfNeuron][numberOfIn];
        for (int neuron = 0; neuron < numberOfNeuron; neuron++) {
            double[] dwn = dw[neuron];
            double cn = c[neuron];
            for (int input = 0; input < numberOfIn; input++) {
                dwn[input] = cn * x[input];
            }
        }
        return dw;
    }


}
