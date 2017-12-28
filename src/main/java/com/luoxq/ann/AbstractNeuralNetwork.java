package com.luoxq.ann;

/**
 * Created by luoxq on 2017/5/21.
 */
public abstract class AbstractNeuralNetwork implements NeuralNetwork {

    public double[] train(double[] in, double[] expect) {
        double[] out = f(in);
        double[] delta = Math.sub(expect, out);
        return train(delta);
    }

}
