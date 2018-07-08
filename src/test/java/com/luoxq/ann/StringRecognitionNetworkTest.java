package com.luoxq.ann;

import org.junit.Ignore;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.Random;

@Ignore
public class StringRecognitionNetworkTest {

    @Test
    public void getImageSize() throws Exception {
        BufferedImage img = ImageIO.read(new File("captcha/valid/valid/2cg7.png"));
        System.out.println("width: " + img.getWidth() + ", height: " + img.getHeight());
    }

    @Test
    public void testToDouble() throws Exception {

        BufferedImage img = ImageIO.read(new File("captcha/shibie/2qxz.png"));
        System.out.println("width: " + img.getWidth() + ", height: " + img.getHeight());

        CharRecognitionNetwork cn = new CharRecognitionNetwork();

        double[] in = cn.toDoubles(img);
    }

    @Test
    public void removeOldFile() throws Exception {
        File dir = new File("captcha/split/");
        String[] names = dir.list();
        for (String s : names) {
            System.out.println(s);
            if (s.matches("[0-9a-z]{1}\\_[0-9]+\\.png")) {
                File file = new File(dir, s);
                System.out.println("Deleting " + file);
                file.delete();
            }
        }
    }

    @Test
    public void pickTestFiles() throws Exception {
        File dir = new File("captcha/split");
        String[] names = dir.list((d, n) -> n.endsWith(".png"));
        System.out.println("Count: " + names.length);
        Arrays.sort(names);
        for (int i = 0; i < names.length; i++) {
            if (i % 10 == 0) {
                String name = names[i];
                new File(dir, name).renameTo(new File(dir.getPath() + "/test/" + name));
            }
        }
    }

    @Test
    public void train() throws Exception {
        StringRecognitionNetwork cn = new StringRecognitionNetwork();
        //CaptchaNetwork cn = loadFromSer();

        cn.loadTrainingData(new File("captcha/valid/valid/"));
        cn.loadTrainingData(new File("captcha/valid2/valid"));
        cn.loadTestData(new File("captcha/valid/valid/test"));

        double rate = cn.test();
        System.out.println("Rate with Test data: " + rate);

        double learningRate = 1;
        cn.setLearningRate(learningRate);
        cn.skipCorrect(false);
        double maxRate = 0;
        Random rand = new Random(System.currentTimeMillis());
        long time = System.currentTimeMillis();
        for (int i = 0; i < 10000000; i++) {
            if (i % 10 == 0) {
//                System.out.println("Cherry Picking: " +
//                        cn.cherryPick(10000));
                cn.shuffleTrainingData();
            }
            cn.train();
            double trainRate = cn.testWithTrainingData();
            double testRate = cn.test();

            if (testRate > maxRate) {
                maxRate = testRate;
                if (testRate > 0.8) {
                    String pathname = "CaptchaNetwork." + testRate + ".ser";
                    System.out.println("Writing: " + pathname);
                    cn.save(new File(pathname));
                }
            }

            System.out.printf("Round %d, Traing: %1.4f, Test: %1.4f, Max: %1.4f, time: %d, learningRate: %1.4f\n", i, trainRate, testRate, maxRate, (System.currentTimeMillis() - time), learningRate);
        }
    }

    @Test
    public void testData() throws Exception {
        StringRecognitionNetwork cn = loadFromSer();
        cn.loadTestData(new File("captcha/valid/valid/test"));
        double rate = cn.test();
        System.out.println("Rate with Test data: " + rate);
    }

    private StringRecognitionNetwork loadFromSer() throws IOException, ClassNotFoundException {
        StringRecognitionNetwork cn = new StringRecognitionNetwork();
        cn.load(new File("StringRecognitionNetwork.0.8782029950083194.ser"));
        return cn;
    }

    @Test
    public void printNetwork() throws Exception {
        ObjectInputStream in = new ObjectInputStream(new FileInputStream("StringRecognitionNetwork.0.8782029950083194.ser"));
        NeuralNetwork nn = (NeuralNetwork) in.readObject();
        System.out.println(nn.toJson());
    }
}
