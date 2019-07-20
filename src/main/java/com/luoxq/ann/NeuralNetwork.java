package com.luoxq.ann;

import java.io.Serializable;

/**
 * 神经网络接口定义。定义一个神经网络所具备的基本操作。
 * <p>
 * Created by luoxq on 2017/5/18.
 */
public interface NeuralNetwork extends Serializable {

    /**
     * 激活函数。即这个神经网络所做的计算功能，对于输入in,计算输出。
     * Activation Function.
     *
     * @param in 输入向量（数组）。数组的长度也就是向量的维数代表这个神经网络接受的输入参数维数。
     * @return 计算的结果向量
     */
    double[] f(double[] in);

    /**
     * 训练函数
     *
     * @param in     输入向量
     * @param expect 期望的输出向量
     * @return 反向传播之后的损失向量，可以用作前置神经网络的损失向量。
     */
    double[] train(double[] in, double[] expect);

    /**
     * 训练函数
     *
     * @param cost 损失向量，即根据某输入向量计算出来的输出向量与期望的输出向量之差值。
     * @return 反向传播之后的损失向量，可以用作前置神经网络的损失向量。
     */
    double[] train(double[] cost);


    /**
     * 学习率 用来控制训练时每次调整的大小。
     */
    double getLearningRate();

    void setLearningRate(double learningRate);

    /**
     * 输入维数。
     */
    int getInputSize();

    int getOutputSize();

    String toJson();
}
