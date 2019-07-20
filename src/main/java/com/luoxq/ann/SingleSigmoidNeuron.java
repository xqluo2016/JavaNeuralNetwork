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
        return x * w + b;
    }

    double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public double[] gradient(double x, double y) {
        double[] g = super.gradient(x, y);
        double dz = dz(y);
        g[0] *= dz;
        g[1] *= dz;
        return g;
    }

    protected double dz(double z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }
}
