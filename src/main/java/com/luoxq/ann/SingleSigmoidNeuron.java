package com.luoxq.ann; /**
 * Created by luoxq on 17/4/9.
 */

public class SingleSigmoidNeuron extends SingleLinearNeuron {

    private static final long serialVersionUID = 1578550361939502383L;

    public SingleSigmoidNeuron(double weight, double bias) {
        super(weight, bias);
    }

    public double f(double x) {
        return sigmoid(z(x));
    }

    double z(double x) {
        return x * weight + bias;
    }

    double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    // c = w*x + b - y
    public double[] gradient(double x, double y) {
        double[] g = super.gradient(x, y);
        double z = z(x);
        double dz = dz(z);
        g[0] *= dz;
        g[1] *= dz;
        return g;
    }

    protected double dz(double z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }

    protected SingleLinearNeuron getTarget() {
        return new SingleSigmoidNeuron(3, 3);
    }


    public static void main(String... args) {
        SingleSigmoidNeuron n = new SingleSigmoidNeuron(0, 0);
        //target: y = 3*x + 3;
        double rate = 5;
        int epoch = 100;
        int trainingSize = 20;
        for (int i = 0; i < epoch; i++) {
            double[][] data = n.generateTrainingData(trainingSize);
            n.train(data, rate);
            System.out.printf("Epoch: %3d,  W: %f, B: %f \n", i, n.weight, n.bias);
        }
    }
}
