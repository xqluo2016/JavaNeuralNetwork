package com.luoxq.ann;

import java.util.Random;

/**
 * 单个线性神经元。
 * <p>
 * 这个神经元的函数是:
 * f(x) = w*x + b
 * <p>
 * 也就是二维坐标系中的一条直线。
 */
public class SingleLinearNeuron {

    private static final long serialVersionUID = 1578550361939502383L;

    double w;
    double b;

    public SingleLinearNeuron(double weight, double bias) {
        this.w = weight;
        this.b = bias;
    }

    /**
     * 激活函数。
     * <p>
     * The activation function.
     */
    public double f(double x) {
        return w * x + b;
    }

    /**
     * 损失函数。对参数x调用激活函数f，并计算结果与期望结果y的差值。
     * <p>
     * cost（loss) function.
     * we use the difference between 'f(x)' and expected 'f(x)' - parameter 'y'.
     */
    public double cost(double x, double y) {
        return f(x) - y;
    }


    public double[] gradient(double x, double y) {
        double c = cost(x, y);
        double sign = c==0? 0: c / Math.abs(c);
        double dw = -x * sign;
        double db = -1 * sign;
        return new double[]{dw, db};
    }

    public void train(double[][] data, double rate) {
        for (int i = 0; i < data.length; i++) {
            double x = data[i][0];
            double y = data[i][1];
            double[] gradient = gradient(x, y);
            w += gradient[0] * rate;
            b += gradient[1] * rate;
        }
    }


}
