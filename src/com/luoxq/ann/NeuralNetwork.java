package com.luoxq.ann;

import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Random;

import static java.lang.Math.exp;
import static com.luoxq.ann.Util.*;

public class NeuralNetwork {
    int[] shape;
    int layers;
    double[][][] weights;
    double[][] bias;
    double[][] zs;
    double[][] xs;

    public NeuralNetwork(int... shape) {
        this.shape = shape;
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

    String toJson() {
        return "{shape:" + Arrays.toString(shape) + ",weights:" + Arrays.deepToString(weights) + ",bias:" + Arrays.deepToString(bias) + "}";
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

    double[] f(double[] in) {
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

    void train(double[] in, double[] expect, double rate) {
        double[] y = f(in);
        double[] cost = sub(expect, y);
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

        weights = add(weights, mul(dw, rate));
        bias = add(bias, mul(db, rate));
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
