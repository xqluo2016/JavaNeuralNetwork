package com.luoxq.ann;

import java.io.Serializable;

/**
 * Created by luoxq on 2017/5/26.
 */
public class DataRecord implements Serializable {
    private static final long serialVersionUID = 1578550361939502383L;

    public String label;
    public double[] input;
    public double[] output;
    public int maxIndex;
    public boolean correct;
}
