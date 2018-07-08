package com.luoxq.ann;

import com.luoxq.ann.app.FontRecognizer;
import org.junit.Ignore;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Created by luoxq on 2017/7/12.
 */
@Ignore
public class FontRecognizerTest {

    @Test
    public void test() {
        FontRecognizer r = new FontRecognizer();
        r.loadData();
        double rate = r.train(100000);
        System.out.println("Correct Rate: " + rate);
    }

    @Test
    public void viewData() throws IOException {
        FontRecognizer r = new FontRecognizer();
        r.loadData();
        DataRecord[] data = r.getData();
        BufferedImage img = new BufferedImage(50, 50 * 100, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < 100 && i < data.length; i++) {
            DataRecord rec = data[i];
            double[] pixels = rec.input;
            for (int x = 0; x < 50; x++) {
                for (int y = 0; y < 50; y++) {
                    img.setRGB(x, i * 50 + y, (int) (pixels[y * 50 + x] * 255));
                }
            }
        }
        File file = File.createTempFile("font", ".png");
        ImageIO.write(img, "png", file);
        System.out.println(file.getAbsolutePath());
    }
}
