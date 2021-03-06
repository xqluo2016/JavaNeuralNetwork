package com.luoxq.ann;

import java.util.Random;

import static java.lang.Math.exp;

/**
 * Created by luoxq on 17/4/16.
 */
public class Math {

    public static double exp(double v) {
        return java.lang.Math.exp(v);
    }

    public static double abs(double v) {
        return v < 0 ? -v : v;
    }

    public static double max(double a, double b) {
        return a > b ? a : b;
    }

    public static double max(double a, double b, double c) {
        return a > b ? (a > c ? a : c) : (b > c ? b : c);
    }

    public static int maxIndex(double[] out, int start, int end) {
        double max = out[start];
        int maxIndex = start;
        for (int i = start + 1; i < end; i++) {
            if (out[i] > max) {
                max = out[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public interface Op {
        double map(double d);
    }

    public static double[] map(double[] v, Op op) {
        double[] r = new double[v.length];
        for (int i = 0; i < r.length; i++) {
            r[i] = op.map(v[i]);
        }
        return r;
    }

    public static double[] range(double start, double end, double step) {
        int size = java.lang.Math.abs((int) ((end - start) / step)) + 1;
        double[] d = new double[size];
        d[0] = start;
        for (int i = 1; i < d.length; i++) {
            d[i] = d[i - 1] + step;
        }
        return d;
    }

    public static double max(double[] d) {
        double max = d[0];
        for (int i = 1; i < d.length; i++) {
            if (max < d[i]) {
                max = d[i];
            }
        }
        return max;
    }

    public static int maxIndex(double[] d) {
        double max = d[0];
        int idx = 0;
        for (int i = 1; i < d.length; i++) {
            if (max < d[i]) {
                max = d[i];
                idx = i;
            }
        }
        return idx;
    }

    public static final Random rand = new Random(System.currentTimeMillis());

    public static void shuffle(Object[] arr) {
        for (int i = 0; i < arr.length; i++) {
            int x = rand.nextInt(arr.length);
            int y = rand.nextInt(arr.length);
            Object o = arr[x];
            arr[x] = arr[y];
            arr[y] = o;
        }
    }

    public static void fillRandom(double[] d) {
        for (int i = 0; i < d.length; i++) {
            d[i] = rand.nextGaussian();
        }
    }

    public static void fillRandom(double[][] d) {
        for (int i = 0; i < d.length; i++) {
            fillRandom(d[i]);
        }
    }

    public static void fillRandom(double[][][] d) {
        for (int i = 0; i < d.length; i++) {
            fillRandom(d[i]);
        }
    }

    public static double sigmoid(double d) {
        return 1.0 / (1.0 + exp(-d));
    }

    public static double[] sigmoid(double[] d) {
        int length = d.length;
        double[] v = new double[length];
        for (int i = 0; i < length; i++) {
            v[i] = sigmoid(d[i]);
        }
        return v;
    }


    public static double[] signmoidPrime(double d[]) {
        int length = d.length;
        double[] v = new double[length];
        for (int i = 0; i < length; i++) {
            v[i] = sigmoidPrime(d[i]);
        }
        return v;
    }

    public static double sigmoidPrime(double d) {
        return sigmoid(d) * (1 - sigmoid(d));
    }

    public static double[] sub(double[] a, double[] b) {
        int len = a.length;
        double[] v = new double[len];
        for (int i = 0; i < len; i++) {
            v[i] = a[i] - b[i];
        }
        return v;
    }

    //V[i]*X[i]
    public static double[] mul(double[] v, double[] x) {
        double[] d = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            d[i] = v[i] * x[i];
        }
        return d;
    }

    public static double[][][] mul(double[][][] a, double b) {
        double[][][] v = new double[a.length][][];
        for (int i = 0; i < a.length; i++) {
            v[i] = mul(a[i], b);
        }
        return v;
    }


    public static double[][] mul(double[][] a, double b) {
        double[][] v = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            v[i] = mul(a[i], b);
        }
        return v;
    }

    public static double[] mul(double[] a, double b) {
        double[] d = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            d[i] = a[i] * b;
        }
        return d;
    }

    public static double[][][] add(double[][][] a, double[][][] b) {
        double[][][] v = new double[a.length][][];
        for (int i = 0; i < a.length; i++) {
            v[i] = add(a[i], b[i]);
        }
        return v;
    }

    public static double[][] add(double[][] a, double[][] b) {
        int length = a.length;
        double[][] v = new double[length][];
        for (int i = 0; i < length; i++) {
            v[i] = add(a[i], b[i]);
        }
        return v;
    }

    public static double[] add(double[] a, double[] b) {
        int length = a.length;
        double[] v = new double[length];
        for (int i = 0; i < length; i++) {
            v[i] = a[i] + b[i];
        }
        return v;
    }

    public static double dot(double[] w, double[] x) {
        double v = 0;
        for (int i = 0; i < w.length; i++) {
            v += w[i] * x[i];
        }
        return v;
    }
}
