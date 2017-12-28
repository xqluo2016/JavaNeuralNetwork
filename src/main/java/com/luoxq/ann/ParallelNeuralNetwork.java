package com.luoxq.ann;

/**
 * Created by luoxq on 2017/5/21.
 */
public class ParallelNeuralNetwork extends AbstractNeuralNetwork implements NeuralNetwork {
    private static final long serialVersionUID = 1578550361939502383L;

    private final NeuralNetwork[] networks;
    private final int inputSize;
    private final int outputSize;

    public ParallelNeuralNetwork(NeuralNetwork... networks) {
        this.networks = networks;
        int inSize = 0;
        int outSize = 0;
        for (NeuralNetwork n : networks) {
            inSize += n.getInputSize();
            outSize += n.getOutputSize();
        }
        this.inputSize = inSize;
        this.outputSize = outSize;
    }

    @Override
    public double[] f(double[] input) {
        double[] output = new double[outputSize];

        int inPos = 0;
        int outPos = 0;

        for (NeuralNetwork n : networks) {
            int inSize = n.getInputSize();
            double[] in = new double[inSize];
            System.arraycopy(input, inPos, in, 0, inSize);

            double[] out = n.f(in);
            int outSize = out.length;
            System.arraycopy(out, 0, output, outPos, outSize);

            inPos += inSize;
            outPos += outSize;
        }
        return output;
    }

    @Override
    public double[] train(double[] cost) {
        double[] delta = new double[inputSize];

        int cpos = 0;
        int dpos = 0;

        for (NeuralNetwork n : networks) {
            int cSize = n.getOutputSize();
            double[] c = new double[cSize];
            System.arraycopy(cost, cpos, c, 0, cSize);

            double[] d = n.train(c);
            int dSize = d.length;
            System.arraycopy(d, 0, delta, dpos, dSize);

            cpos += cSize;
            dpos += dSize;
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
        return inputSize;
    }

    @Override
    public int getOutputSize() {
        return outputSize;
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
