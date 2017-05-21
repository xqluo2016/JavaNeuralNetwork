package com.luoxq.ann;

import java.io.Serializable;

/**
 * Created by luoxq on 2017/5/18.
 */
public interface NeuralNetwork extends Serializable {
    double[] f(double[] in);

    double[] train(double[] in, double[] expect);

    double[] train(double[] cost);

    double getLearningRate();

    void setLearningRate(double learningRate);

    int getInputSize();

    int getOutputSize();

    public String toJson();
}
