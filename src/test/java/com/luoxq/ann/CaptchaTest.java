package com.luoxq.ann;

import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

public class CaptchaTest {

    @Test
    public void getImageSize() throws Exception {
        BufferedImage img = ImageIO.read(new File("captcha/shibie/2qxz.png"));
        System.out.println("width: " + img.getWidth() + ", height: " + img.getHeight());
    }

    @Test
    public void testToDouble() throws Exception {

        BufferedImage img = ImageIO.read(new File("captcha/shibie/2qxz.png"));
        System.out.println("width: " + img.getWidth() + ", height: " + img.getHeight());

        CaptchaNetwork cn = new CaptchaNetwork();

        double[] in = cn.toDoubles(img);
    }

    @Test
    public void train() throws Exception {
        CaptchaNetwork cn = new CaptchaNetwork();
        cn.loadTrainingData();
        cn.loadTestData();
        cn.setLearningRate(0.5);
        double maxRate = 0;
        for (int i = 0; i < 100000; i++) {
            cn.train();
            double trainRate = cn.testWithTrainingData();
            double testRate = cn.test();
            System.out.printf("%1.2f, %1.2f\n", trainRate, testRate);
            if (testRate > maxRate) {
                maxRate = testRate;
                if (testRate > 0.9) {
                    cn.save(new File("CaptchaNetwork." + testRate + ".ser"));
                }
            }
            cn.shuffleTrainingData();
        }
    }

    @Test
    public void testData() throws Exception {
        CaptchaNetwork cn = new CaptchaNetwork();
        cn.load(new File("CaptchaNetwork.0.90625.ser"));
        cn.loadTestData();
        double rate = cn.test();
        System.out.println("Rate with Test data: " + rate);
    }
}
