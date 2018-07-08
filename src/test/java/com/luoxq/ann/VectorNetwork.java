package com.luoxq.ann;

import org.junit.Ignore;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

@Ignore
public class VectorNetwork {

    @Test
    public void test() throws Exception {
        BufferedImage img = ImageIO.read(new File("captcha/valid/split/2_2dwj_2.png"));
        int width = img.getWidth();
        int height = img.getHeight();

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {

            }
        }

    }
}
