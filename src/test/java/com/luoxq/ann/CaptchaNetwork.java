package com.luoxq.ann;

import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;

public class CaptchaNetwork {


    NeuralNetwork nn = new SigmoidNeuralNetwork(30 * 30, 30, 30, 36);

    List<DataRecord> trainingData = new ArrayList<>();
    List<DataRecord> testData = new ArrayList<>();

    static final String chars = "0123456789abcdefghijklmnopqrstuvwxyz";

    @Test
    public void loadTrainingData() throws IOException {
        File dir = new File("captcha/split");
        load(dir, trainingData);
    }

    @Test
    public void loadTestData() throws IOException {
        File dir = new File("captcha/split/test");
        List<DataRecord> data = this.testData;
        load(dir, data);
    }

    private void load(File dir, List<DataRecord> data) throws IOException {
        for (File f : dir.listFiles()) {
            if (f.getName().endsWith(".png")) {
                char txt = Character.toLowerCase(f.getName().charAt(0));
                BufferedImage img = ImageIO.read(f);
                double[] in = toDoubles(img);
                double[] out = new double[36];
                int index = chars.indexOf(txt);
                out[index] = 1.0;
                DataRecord record = new DataRecord();
                record.input = in;
                record.output = out;
                record.label = "" + txt;
                record.maxIndex = index;
                data.add(record);
                printRecord(record);
            }
        }
        System.out.println("Loaded record " + data.size());
    }

    public void printRecord(DataRecord dr) {
        double[] in = dr.input;
        StringBuilder sb = new StringBuilder("--------").append(dr.label).append("-").append(dr.maxIndex).append("---------\n")
                .append(Arrays.toString(dr.output)).append("\n");
        for (int y = 0; y < 30; y++) {
            for (int x = 0; x < 30; x++) {
                sb.append(in[y * 30 + x] < 0.5 ? '0' : ' ');
            }
            sb.append("\n");
        }
        System.out.println(sb);
    }

    double[] toDoubles(BufferedImage img) {
        int[] rgb = img.getRGB(0, 0, img.getWidth(), img.getHeight(), null, 0, img.getWidth());
        double[] data = new double[rgb.length];
        for (int i = 0; i < rgb.length; i++) {
            int c = rgb[i];
            int v = (c >> 16) & 0xff + (c >> 8) & 0xff + c & 0xff;
            data[i] = v / 255.0;
        }
        return data;
    }

    public char recognize(BufferedImage img) {
        double[] in = toDoubles(img);
        return recognize(in);
    }

    private char recognize(double[] in) {
        double[] out = nn.f(in);
        return chars.charAt(Util.maxIndex(out));
    }

    public void train() {
        for (DataRecord r : trainingData) {
            if (!r.correct)
                nn.train(r.input, r.output);
        }
    }

    public void shuffleTrainingData() {
        DataRecord[] array = new DataRecord[trainingData.size()];
        array = trainingData.toArray(array);
        Random rand = new Random(System.currentTimeMillis());
        for (int i = 0; i < trainingData.size(); i++) {
            int a = rand.nextInt(trainingData.size());
            int b = rand.nextInt(trainingData.size());
            DataRecord tmp = array[a];
            array[a] = array[b];
            array[b] = tmp;
        }
    }

    public double testWithTrainingData() {
        List<DataRecord> data = this.trainingData;
        return test(data);
    }

    public double test() {
        return test(testData);
    }

    private double test(List<DataRecord> data) {
        int correct = 0;
        int wrong = 0;
        for (DataRecord r : data) {
            char c = recognize(r.input);
            char expect = chars.charAt(r.maxIndex);
            if (c == expect) {
                r.correct = true;
                correct++;
            } else {
                r.correct = false;
                wrong++;
            }
        }

        double rate = correct * 1.0 / (correct + wrong);
        return rate;
    }

    public void setLearningRate(double learningRate) {
        nn.setLearningRate(learningRate);
    }

    public void save(File file) throws IOException {
        ObjectOutput out = new ObjectOutputStream(new FileOutputStream(file));
        out.writeObject(nn);
        out.close();
    }

    public void load(File file) throws IOException, ClassNotFoundException {
        ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
        nn = (NeuralNetwork) in.readObject();
    }
}
