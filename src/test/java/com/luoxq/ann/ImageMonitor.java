package com.luoxq.ann;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageMonitor extends JFrame {

    private BufferedImage image;

    private final JPanel canvas;

    public static void main(String... args) throws IOException {
        ImageMonitor monitor = new ImageMonitor(ImageIO.read(new File("captcha/valid/split/3_mks3_409.png")));
    }

    public ImageMonitor(BufferedImage img) throws HeadlessException {
        super("ImageMonitor");
        image = img;
        setLayout(new BorderLayout());
        canvas = new JPanel() {
            public void paint(Graphics g) {
                if (image != null) {
                    g.drawImage(image, 0, 0, ImageMonitor.this.getWidth(),
                            image.getHeight() * ImageMonitor.this.getWidth() / image.getWidth(), null);
                }
            }
        };

        add(canvas, BorderLayout.CENTER);

        setSize(500, 500);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setVisible(true);
        new Thread(
                () -> {
                    while (ImageMonitor.this.isVisible()) {
                        try {
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                        canvas.repaint();
                    }
                }
        ).start();
    }

}
