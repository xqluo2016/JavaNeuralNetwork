package com.luoxq.ann;

import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class FontPositionTest {

    static class KeyImage {
        double[] input;
        double[] output;
        List<Point> points;
    }

    String s = loadChineseChars();
    Random rand = new Random();


    @Test
    public void generateImage() throws Exception {
        BufferedImage img = generateKeyImage();
        ImageIO.write(img, "jpeg", new File("chinese.jpg"));
    }

    @Test
    public void train() throws Exception {
        List<KeyImage> trainingData = new ArrayList<KeyImage>();
        for (int i = 0; i < 10000; i++) {

        }
    }


    public BufferedImage generateKeyImage() {
        BufferedImage img = new BufferedImage(116, 40, BufferedImage.TYPE_BYTE_GRAY);
        int fontCount = rand.nextInt(3) + 3;
        Graphics g = img.getGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, img.getWidth(), img.getHeight());
        int width = 116 / fontCount;
        for (int i = 0; i < fontCount; i++) {
            char ch = s.charAt(rand.nextInt(s.length()));
            g.setColor(Color.BLACK);
            g.setFont(new Font("", 0, 20));

            int x = i * width + rand.nextInt(10);
            int y = 20 + rand.nextInt(20);

            g.drawChars(new char[]{ch}, 0, 1, x, y);
        }
        return img;
    }

    public String loadChineseChars() {
        try {
            BufferedReader chars = new BufferedReader(new InputStreamReader(getClass().getResourceAsStream("/chinese_surname.txt")));
            StringBuilder buf = new StringBuilder();
            for (String line = null; (line = chars.readLine()) != null; ) {
                buf.append(line);
            }
            return buf.toString();
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}
