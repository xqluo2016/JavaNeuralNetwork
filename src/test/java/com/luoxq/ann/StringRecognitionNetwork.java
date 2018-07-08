package com.luoxq.ann;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class StringRecognitionNetwork implements Serializable {


    NeuralNetwork nn;

    final List<DataRecord> allTrainingData = new ArrayList<>();
    List<DataRecord> trainingData = allTrainingData;
    List<DataRecord> testData = new ArrayList<>();

    private boolean skipCorrect = false;

    static final String chars = "0123456789abcdefghijklmnopqrstuvwxyz";


    public StringRecognitionNetwork(NeuralNetwork nn) {
        this.nn = nn;
    }

    public StringRecognitionNetwork() {
        nn = new SigmoidNeuralNetwork(160 * 70, 100, 100, 36 * 4);
    }

    public void loadTrainingData(File dir) throws IOException {
        loadDir(dir, allTrainingData);
        System.out.println("Loaded training record " + allTrainingData.size());
    }

    public void loadTestData(File dir) throws IOException {
        List<DataRecord> data = this.testData;
        loadDir(dir, data);
        System.out.println("Loaded test record " + data.size());
    }

    private void loadDir(File dir, List<DataRecord> data) throws IOException {
        for (File f : dir.listFiles()) {
            if (f.getName().endsWith(".png")) {
                BufferedImage img = ImageIO.read(f);
                loadRecord(data, f.getName().substring(0, 4), img);
            }
        }
    }

    public void loadTrainingRecord(String ch, BufferedImage img) {
        loadRecord(this.allTrainingData, ch, img);
    }

    public void loadTestRecord(String ch, BufferedImage img) {
        loadRecord(this.testData, ch, img);
    }

    private void loadRecord(List<DataRecord> data, String ch, BufferedImage img) {
        ch = ch.toLowerCase();
        double[] in = toDoubles(img);
        double[] out = new double[36 * 4];
        for (int i = 0; i < 4; i++) {
            int index = chars.indexOf(ch.charAt(i));
            out[index] = 1.0;
        }
        DataRecord record = new DataRecord();
        record.input = in;
        record.output = out;
        record.label = ch;
        data.add(record);
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

    public String recognize(BufferedImage img) {
        double[] in = toDoubles(img);
        return recognize(in);
    }

    private String recognize(double[] in) {
        double[] out = nn.call(in);
        char[] ch = new char[4];
        for (int i = 0; i < 4; i++) {
            ch[i] = chars.charAt(Math.maxIndex(out, i * 36, (i + 1) * 36) - i * 36);
        }
        return new String(ch);
    }

    public void train() {
        for (DataRecord r : trainingData) {
            if (r.correct && skipCorrect) {
            } else {
                nn.train(r.input, r.output);
            }
        }
    }

    public int cherryPick(int count) {
        if (allTrainingData != null && allTrainingData.size() > 0) {
            trainingData = new ArrayList<>();
            Random rand = new Random(System.currentTimeMillis());
            for (int i = 0; i < count; i++) {
                trainingData.add(allTrainingData.get(rand.nextInt(allTrainingData.size())));
            }
            return count;
        }
        return 0;
    }

    public void shuffleTrainingData() {
        List<DataRecord> trainingData = this.allTrainingData;
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
            String c = recognize(r.input);
            String expect = r.label;
            if (c.equals(expect)) {
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

    public int getAllTrainingDataSize() {
        return allTrainingData.size();
    }

    public void skipCorrect(boolean skip) {
        this.skipCorrect = skip;
    }

    public boolean isSkipCorrect() {
        return this.skipCorrect;
    }
}
