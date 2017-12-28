package com.luoxq.ann;

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
    public double[] call(double[] in) {
        double[] out = in;
        for (NeuralNetwork n : networks) {
            out = n.call(out);
        }
        return out;
    }

    @Override
    public double[] train(double[] in, double[] expect) {
        double[] out = call(in);
        double[] delta = Math.sub(expect, out);
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
