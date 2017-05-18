package com.luoxq.ann;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Created by luoxq on 2017/4/22.
 */
public class FontRecognizer {

    public static class Data {
        public double[] data;
        public double[] output;
        public char value;
    }

    public static void main(String... args) {
        FontRecognizer r = new FontRecognizer();
        r.loadData();
        r.train();
    }

    private List<Data> data;
    int[] shape = {20 * 15, 20, 15, 95};
    SigmoidNetwork nn = new SigmoidNetwork(shape);

    public int test() {
        int correct = 0;
        for (Data d : data) {
            if (Util.maxIndex(nn.f(d.data)) == d.value - 32) {
                correct++;
            }
        }
        return correct;
    }

    public void train() {
        System.out.println("Shape: " + Arrays.toString(shape));
        System.out.println("Data Size: " + data.size());
        double rate = 2;
        System.out.println("Rate:" + rate);
        if (data.size() == 0) {
            return;
        }
        double correct = test() * 1.0 / data.size();
        System.out.println("Initial correct rate:" + correct);
        long time = System.currentTimeMillis();
        for (int i = 0; i < 100000000; i++) {
            for (Data d : data) {
                nn.train(d.data, d.output, rate);
            }

            if (i < 10 || i % 10 == 0) {
                double c = test() * 1.0 / data.size();
                if (c > correct) {
                    correct = c;
                    System.out.println(i + ",\t" + (System.currentTimeMillis() - time) / 1000 + ",\t" + correct);
                }
                if (c > 0.999) {
                    break;
                }
            }
        }
    }

    public List<Data> getData() {
        return Collections.unmodifiableList(data);
    }

    public void loadData() {
        List<Data> data = new ArrayList<Data>();
        Font[] allFonts = GraphicsEnvironment.getLocalGraphicsEnvironment().getAllFonts();
        final int limit = allFonts.length;
        for (int f = 0; f < limit; f++) {
            Font font = allFonts[f];
            for (char c = 32; c < 127; c++) {
                BufferedImage img = new BufferedImage(15, 20, BufferedImage.TYPE_BYTE_GRAY);
                Graphics2D g = (Graphics2D) img.getGraphics();
                g.drawString(c + "", 2, 15);
                double[] d = new double[20 * 15];
                for (int x = 0; x < 15; x++) {
                    for (int y = 0; y < 20; y++) {
                        d[15 * y + x] = img.getRGB(x, y) & 0xff;
                    }
                }
                Data record = new Data();
                record.data = d;
                record.value = c;
                record.output = new double[95];
                record.output[c - 32] = 1;
                data.add(record);
            }
        }
        this.data = data;
    }


}
