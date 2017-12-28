package com.luoxq.ann;

import com.luoxq.ann.NeuralNetwork;
import com.luoxq.ann.Util;

/**
 * Created by luoxq on 2017/5/21.
 */
public class LinkedNeuralNetwork implements NeuralNetwork {
    private static final long serialVersionUID = 1578550361939502383L;

    NeuralNetwork[] networks;

    public LinkedNeuralNetwork(NeuralNetwork... networks) {
        this.networks = networks;
    }

    @Override
    public double[] f(double[] in) {
        double[] out = in;
        for (NeuralNetwork n : networks) {
            out = n.f(out);
        }
        return out;
    }

    @Override
    public double[] train(double[] in, double[] expect) {
        double[] out = f(in);
        double[] delta = Util.sub(expect, out);
        return train(delta);
    }

    @Override
    public double[] train(double[] cost) {
        double[] delta = cost;
        for (int i = networks.length - 1; i >= 0; i--) {
            NeuralNetwork n = networks[i];
            delta = n.train(delta);
        }
        return delta;
    }

    @Override
    public double getLearningRate() {
        return networks[0].getLearningRate();
    }

    @Override
    public void setLearningRate(double learningRate) {
        for (NeuralNetwork n : networks) {
            n.setLearningRate(learningRate);
        }
    }

    @Override
    public int getInputSize() {
        return networks[0].getInputSize();
    }

    @Override
    public int getOutputSize() {
        return networks[networks.length - 1].getOutputSize();
    }

    @Override
    public String toJson() {
        StringBuilder buf = new StringBuilder("{class: '" + this.getClass().getName() + "', networks:[");
        for (NeuralNetwork n : networks) {
            buf.append(n.toJson()).append(",");
        }
        if (buf.charAt(buf.length() - 1) == ',') {
            buf.setLength(buf.length() - 1);
        }
        buf.append("]}");
        return buf.toString();
    }
}
