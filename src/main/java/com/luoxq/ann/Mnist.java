package com.luoxq.ann;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.Random;
import java.util.zip.GZIPInputStream;

/**
 * Created by luoxq on 17/4/15.
 */
public class Mnist {


    static class Data {
        public byte[] data;
        public int label;
        public double[] input;
        public double[] output;
    }

    public static void main(String... args) throws Exception {
        Mnist mnist = new Mnist();
        mnist.load();
        System.out.println("Data loaded.");
        Random rand = new Random(System.nanoTime());
        for (int i = 0; i < 20; i++) {
            int idx = rand.nextInt(60000);
            Data d = mnist.getTrainingData(idx);
            BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            for (int x = 0; x < 28; x++) {
                for (int y = 0; y < 28; y++) {
                    img.setRGB(x, y, toRgb(d.data[y * 28 + x]));
                }
            }
            File output = new File(i + "_" + d.label + ".png");
            if (!output.exists()) {
                output.createNewFile();
            }
            ImageIO.write(img, "png", output);
        }
    }

    static int toRgb(byte bb) {
        int b = (255 - (0xff & bb));
        return (b << 16 | b << 8 | b) & 0xffffff;
    }


    Data[] trainingSet;
    Data[] testSet;

    public void shuffle() {
        Random rand = new Random();
        for (int i = 0; i < trainingSet.length; i++) {
            int x = rand.nextInt(trainingSet.length);
            Data d = trainingSet[i];
            trainingSet[i] = trainingSet[x];
            trainingSet[x] = trainingSet[i];
        }
    }

    public Data getTrainingData(int idx) {
        return trainingSet[idx];
    }

    public Data[] getTrainingSlice(int start, int count) {
        Data[] ret = new Data[count];
        System.arraycopy(trainingSet, start, ret, 0, count);
        return ret;
    }

    public Data getTestData(int idx) {
        return testSet[idx];
    }

    public Data[] getTestSlice(int start, int count) {
        Data[] ret = new Data[count];
        System.arraycopy(testSet, start, ret, 0, count);
        return ret;
    }


    public void load() {
        trainingSet = load("/train-images-idx3-ubyte.gz", "/train-labels-idx1-ubyte.gz");
        testSet = load("/t10k-images-idx3-ubyte.gz", "/t10k-labels-idx1-ubyte.gz");
        if (trainingSet.length != 60000 || testSet.length != 10000) {
            throw new RuntimeException("Unexpected training/test data size: " + trainingSet.length + "/" + testSet.length);
        }
    }

    private Data[] load(String imgFile, String labelFile) {
        byte[][] images = loadImages(imgFile);
        byte[] labels = loadLabels(labelFile);
        if (images.length != labels.length) {
            throw new RuntimeException("Images and label doesn't match: " + imgFile + " " + labelFile);
        }
        int len = images.length;
        Data[] data = new Data[len];
        for (int i = 0; i < len; i++) {
            data[i] = new Data();
            data[i].data = images[i];
            data[i].label = 0xff & labels[i];
            data[i].input = dataToInput(images[i]);
            data[i].output = labelToOutput(labels[i]);
        }
        return data;
    }

    private double[] labelToOutput(byte label) {
        double[] o = new double[10];
        o[label] = 1;
        return o;
    }

    private double[] dataToInput(byte[] b) {
        double[] d = new double[b.length];
        for (int i = 0; i < b.length; i++) {
            d[i] = (b[i] & 0xff) / 255.0;
        }
        return d;
    }

    private byte[][] loadImages(String imgFile) {
        try (DataInputStream in = new DataInputStream(new GZIPInputStream(this.getClass().getResourceAsStream(imgFile)));) {
            int magic = in.readInt();
            if (magic != 0x00000803) {
                throw new RuntimeException("wrong magic: 0x" + Integer.toHexString(magic));
            }
            int count = in.readInt();
            int rows = in.readInt();
            int cols = in.readInt();
            if (rows != 28 || cols != 28) {
                throw new RuntimeException("Unexpected row and col count: " + rows + "x" + cols);
            }
            byte[][] data = new byte[count][rows * cols];
            for (int i = 0; i < count; i++) {
                in.readFully(data[i]);
            }
            return data;
        } catch (Exception ex) {
            throw new RuntimeException("Failed to read file: " + imgFile, ex);
        }
    }

    private byte[] loadLabels(String labelFile) {
        try (DataInputStream in = new DataInputStream(new GZIPInputStream(this.getClass().getResourceAsStream(labelFile)));) {
            int magic = in.readInt();
            if (magic != 0x00000801) {
                throw new RuntimeException("wrong magic: 0x" + Integer.toHexString(magic));
            }
            int count = in.readInt();
            byte[] data = new byte[count];
            in.readFully(data);
            return data;
        } catch (Exception ex) {
            throw new RuntimeException("Failed to read file: " + labelFile, ex);
        }
    }

}
