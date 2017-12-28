package com.luoxq.ann.app;

import com.luoxq.ann.DataRecord;
import com.luoxq.ann.SigmoidNeuralNetwork;
import com.luoxq.ann.Util;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by luoxq on 2017/4/22.
 */
public class FontRecognizer {
    private static int WIDTH = 50;
    private static int HEIGHT = 50;
    private DataRecord[] data;
    int[] shape = {WIDTH * HEIGHT, 50, 20, 95};
    SigmoidNeuralNetwork nn = new SigmoidNeuralNetwork(shape);

    public int test(boolean print) {
        StringBuilder sb = new StringBuilder();
        int correct = 0;
        for (DataRecord d : data) {
            int maxIndex = Util.maxIndex(nn.f(d.input));
            if (maxIndex == d.maxIndex) {
                correct++;
                d.correct = true;
            } else {
                d.correct = false;
                if (print) {
                    sb.append((char) (d.maxIndex + 32)).append("->").append((char) (maxIndex + 32)).append("\t");
                }
            }
        }
        if (print) {
            System.out.println(sb);
        }
        return correct;
    }

    public double train(int epoches) {
        System.out.println("Shape: " + Arrays.toString(shape));
        System.out.println("DataRecord Size: " + data.length);
        double rate = 2.0;
        nn.setLearningRate(rate);
        System.out.println("Rate:" + rate);
        if (data == null || data.length == 0) {
            return 0;
        }
        double correct = test(false) * 1.0 / data.length;
        System.out.println("Initial correct rate:" + correct);
        long time = System.currentTimeMillis();
        for (int i = 0; i < epoches; i++) {
            Util.shuffle(data);
            for (DataRecord d : data) {
                if (!d.correct)
                    nn.train(d.input, d.output);
            }

            double c = test(false) * 1.0 / data.length;
            if (c != correct) {
                correct = c;
                System.out.println(i + ",\t" + (System.currentTimeMillis() - time) / 1000 + ",\t" + correct);
                if (c > 0.95) {
                    try (ObjectOutputStream out = new ObjectOutputStream(
                            new FileOutputStream("Font." + nn.getClass().getName() + "." + c + ".ser"))) {
                        out.writeObject(nn);
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                    test(true);
                }
            }
        }
        return correct;
    }

    public DataRecord[] getData() {
        return data;
    }

    public void loadData() {
        List<DataRecord> data = new ArrayList<DataRecord>();
        Font[] allFonts = GraphicsEnvironment.getLocalGraphicsEnvironment().getAllFonts();
        final int limit = allFonts.length;
        for (int f = 0; f < 1; f++) {
            Font font = allFonts[f];
            for (char c = 32; c < 127; c++) {
                BufferedImage img = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
                Graphics2D g = (Graphics2D) img.getGraphics();
                g.setFont(new Font(null, 0, HEIGHT / 2));
                g.drawString(c + "", 1, HEIGHT * 2 / 3);
                double[] d = new double[HEIGHT * WIDTH];
                for (int x = 0; x < WIDTH; x++) {
                    for (int y = 0; y < HEIGHT; y++) {
                        d[WIDTH * y + x] = img.getRGB(x, y) & 0xff;
                    }
                }
                DataRecord record = new DataRecord();
                record.input = d;
                record.label = Character.toString(c);
                record.output = new double[95];
                record.output[c - 32] = 1;
                record.maxIndex = c - 32;
                data.add(record);
            }
        }
        this.data = data.toArray(new DataRecord[data.size()]);
    }


}
