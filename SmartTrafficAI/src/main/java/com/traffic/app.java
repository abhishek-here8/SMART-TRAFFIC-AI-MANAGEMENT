
package com.traffic;

import ai.onnxruntime.*;
import nu.pattern.OpenCV;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.swing.*;
import javax.swing.border.EmptyBorder;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.image.BufferedImage;

import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class App {

    // ================ CONFIGURATION ================

    private static final int CAMERA_INDEX = 0;
    private static final int INPUT_SIZE = 640;

    private static final float CONFIDENCE_THRESHOLD = 0.25f;
    private static final float NMS_THRESHOLD = 0.45f;

    private static final int SIGNAL_DURATION = 10;

    private static final String MODEL_PATH = "yolov8n.onnx";

    // COCO class IDs
    // 2 = Car
    // 3 = Motorcycle
    // 5 = Bus
    // 7 = Truck

    private static final int[] VEHICLE_IDS = {2, 3, 5, 7};

    private static final String[] VEHICLE_NAMES = {
        "CAR", "MOTORCYCLE", "BUS", "TRUCK"
    };

    private static final String[] ZONE_NAMES = {
        "NORTH", "EAST", "SOUTH", "WEST"
    };

    private static final Scalar[] ZONE_COLORS = {
        new Scalar(255, 100, 0),
        new Scalar(0, 200, 255),
        new Scalar(0, 255, 100),
        new Scalar(255, 0, 200)
    };

    // ================ VARIABLES ================

    private OrtEnvironment environment;
    private OrtSession session;
    private VideoCapture camera;

    private JFrame window;

    private JLabel cameraLabel;
    private JLabel statusLabel;
    private JLabel totalLabel;
    private JLabel signalLabel;
    private JLabel timerLabel;

    private JLabel[] zoneLabels = new JLabel[4];

    private volatile boolean running = false;

    private volatile int activeSignal = 0;
    private volatile int remainingSeconds = 10;

    private final int[] zoneCounts = new int[4];

    private long signalStartTime;

    private Thread cameraThread;

    private boolean outputInfoPrinted = false;

    // ================ DETECTION CLASS ================

    private static class Detection {

        int classId;
        float confidence;
        Rect box;

        Detection(int classId, float confidence, Rect box) {
            this.classId = classId;
            this.confidence = confidence;
            this.box = box;
        }
    }

    // ================ MAIN ================

    public static void main(String[] args) {

        SwingUtilities.invokeLater(() -> {
            App app = new App();
            app.initialize();
        });
    }

    // ================ INITIALIZE ================

    private void initialize() {

        try {

            OpenCV.loadLocally();

            System.out.println("OpenCV Loaded Successfully!");

            loadYOLOModel();

            createDashboard();

            connectLaptopCamera();

        } catch (Exception e) {

            e.printStackTrace();

            JOptionPane.showMessageDialog(
                null,
                e.getMessage(),
                "SMART TRAFFIC AI ERROR",
                JOptionPane.ERROR_MESSAGE
            );

            shutdown();
        }
    }

    // ================ LOAD YOLO MODEL ================

    private void loadYOLOModel() throws Exception {

        Path modelPath = Path.of(MODEL_PATH);

        if (!Files.exists(modelPath)) {

            throw new Exception(
                "YOLO model not found!\n\n" +
                "Place yolov8n.onnx in the project root folder."
            );
        }

        environment = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions options =
            new OrtSession.SessionOptions();

        options.setOptimizationLevel(
            OrtSession.SessionOptions.OptLevel.ALL_OPT
        );

        session = environment.createSession(
            modelPath.toAbsolutePath().toString(),
            options
        );

        options.close();

        System.out.println("YOLOv8n Model Loaded Successfully!");

        System.out.println(
            "Model Inputs: " + session.getInputInfo().keySet()
        );

        System.out.println(
            "Model Outputs: " + session.getOutputInfo().keySet()
        );
    }

    // ================ CREATE GUI ================

    private void createDashboard() {

        window = new JFrame("SMART TRAFFIC AI - LIVE SYSTEM");

        window.setSize(1300, 800);
        window.setMinimumSize(new Dimension(1000, 650));
        window.setLocationRelativeTo(null);

        window.setDefaultCloseOperation(
            JFrame.DO_NOTHING_ON_CLOSE
        );

        window.setLayout(new BorderLayout(10, 10));

        window.getContentPane().setBackground(
            new Color(15, 23, 42)
        );

        // HEADER

        JLabel heading = new JLabel(
            "SMART TRAFFIC AI",
            SwingConstants.CENTER
        );

        heading.setFont(new Font("Arial", Font.BOLD, 30));
        heading.setForeground(Color.WHITE);
        heading.setBorder(new EmptyBorder(15, 5, 15, 5));

        window.add(heading, BorderLayout.NORTH);

        // CAMERA PANEL

        cameraLabel = new JLabel(
            "CONNECTING TO LAPTOP CAMERA...",
            SwingConstants.CENTER
        );

        cameraLabel.setOpaque(true);
        cameraLabel.setBackground(Color.BLACK);
        cameraLabel.setForeground(Color.WHITE);

        cameraLabel.setFont(
            new Font("Arial", Font.BOLD, 20)
        );

        JPanel videoPanel = new JPanel(new BorderLayout());

        videoPanel.setBackground(Color.BLACK);

        videoPanel.setBorder(
            BorderFactory.createLineBorder(
                new Color(59, 130, 246), 2
            )
        );

        videoPanel.add(cameraLabel, BorderLayout.CENTER);

        window.add(videoPanel, BorderLayout.CENTER);

        // RIGHT DASHBOARD

        JPanel dashboard = new JPanel();

        dashboard.setPreferredSize(new Dimension(285, 0));

        dashboard.setLayout(
            new BoxLayout(dashboard, BoxLayout.Y_AXIS)
        );

        dashboard.setBackground(new Color(30, 41, 59));

        dashboard.setBorder(
            new EmptyBorder(20, 15, 20, 15)
        );

        dashboard.add(createLabel("TRAFFIC DASHBOARD", 19));
        dashboard.add(Box.createVerticalStrut(30));

        totalLabel = createLabel("TOTAL VEHICLES: 0", 17);
        dashboard.add(totalLabel);

        dashboard.add(Box.createVerticalStrut(30));
        dashboard.add(createLabel("FOUR ROAD ZONES", 18));

        for (int i = 0; i < 4; i++) {

            zoneLabels[i] = createLabel(
                ZONE_NAMES[i] + ": 0", 16
            );

            zoneLabels[i].setForeground(
                scalarToColor(ZONE_COLORS[i])
            );

            dashboard.add(Box.createVerticalStrut(18));
            dashboard.add(zoneLabels[i]);
        }

        dashboard.add(Box.createVerticalStrut(35));
        dashboard.add(createLabel("TRAFFIC SIGNAL", 18));

        signalLabel = createLabel("NORTH GREEN", 21);
        signalLabel.setForeground(Color.GREEN);

        dashboard.add(Box.createVerticalStrut(15));
        dashboard.add(signalLabel);

        timerLabel = createLabel("10 SECONDS", 17);

        dashboard.add(Box.createVerticalStrut(10));
        dashboard.add(timerLabel);

        dashboard.add(Box.createVerticalStrut(35));

        statusLabel = createLabel("CAMERA: WAITING", 14);

        dashboard.add(statusLabel);

        window.add(dashboard, BorderLayout.EAST);

        // FOOTER

        JLabel footer = new JLabel(
            "YOLOv8 | OPENCV | ONNX RUNTIME | JAVA",
            SwingConstants.CENTER
        );

        footer.setForeground(Color.LIGHT_GRAY);
        footer.setBorder(new EmptyBorder(12, 5, 12, 5));

        window.add(footer, BorderLayout.SOUTH);

        window.addWindowListener(
            new java.awt.event.WindowAdapter() {

                @Override
                public void windowClosing(
                    java.awt.event.WindowEvent e
                ) {
                    shutdown();
                }
            }
        );

        window.setVisible(true);

        // SIGNAL TIMER

        signalStartTime = System.currentTimeMillis();

        javax.swing.Timer timer = new javax.swing.Timer(
            200,
            e -> updateSignal()
        );

        timer.start();
    }

    private JLabel createLabel(String text, int size) {

        JLabel label = new JLabel(text);

        label.setFont(
            new Font("Arial", Font.BOLD, size)
        );

        label.setForeground(Color.WHITE);

        label.setAlignmentX(
            java.awt.Component.LEFT_ALIGNMENT
        );

        return label;
    }

    private Color scalarToColor(Scalar scalar) {

        return new Color(
            (int) scalar.val[2],
            (int) scalar.val[1],
            (int) scalar.val[0]
        );
    }

    // ================ LAPTOP CAMERA ================

    private void connectLaptopCamera() {

        camera = new VideoCapture();

        boolean connected = camera.open(
            CAMERA_INDEX,
            Videoio.CAP_DSHOW
        );

        if (!connected || !camera.isOpened()) {

            camera.release();

            camera = new VideoCapture();

            connected = camera.open(CAMERA_INDEX);
        }

        if (!connected || !camera.isOpened()) {

            statusLabel.setText("CAMERA CONNECTION FAILED");
            statusLabel.setForeground(Color.RED);

            cameraLabel.setText(
                "LAPTOP CAMERA NOT FOUND"
            );

            System.out.println(
                "Could not connect to laptop webcam."
            );

            return;
        }

        camera.set(
            Videoio.CAP_PROP_FRAME_WIDTH, 1280
        );

        camera.set(
            Videoio.CAP_PROP_FRAME_HEIGHT, 720
        );

        camera.set(
            Videoio.CAP_PROP_FPS, 30
        );

        running = true;

        statusLabel.setText("LAPTOP CAMERA CONNECTED");
        statusLabel.setForeground(Color.GREEN);

        cameraLabel.setText(
            "CAMERA CONNECTED - YOLO STARTING"
        );

        System.out.println(
            "Laptop Webcam Connected Successfully!"
        );

        cameraThread = new Thread(
            this::processCamera,
            "Traffic-Camera-Thread"
        );

        cameraThread.setDaemon(true);
        cameraThread.start();
    }

    // ================ CAMERA PROCESSING ================

    private void processCamera() {

        Mat frame = new Mat();

        while (running) {

            try {

                if (!camera.read(frame) || frame.empty()) {

                    Thread.sleep(30);
                    continue;
                }

                List<Detection> detections =
                    detectVehicles(frame);

                Arrays.fill(zoneCounts, 0);

                for (Detection detection : detections) {

                    int centerX =
                        detection.box.x +
                        detection.box.width / 2;

                    int centerY =
                        detection.box.y +
                        detection.box.height / 2;

                    int zone = getZone(
                        centerX,
                        centerY,
                        frame.cols(),
                        frame.rows()
                    );

                    zoneCounts[zone]++;

                    drawDetection(
                        frame,
                        detection,
                        zone
                    );
                }

                drawZones(frame);
                drawSignalOnCamera(frame);

                updateDashboard();
                displayFrame(frame);

            } catch (InterruptedException e) {

                Thread.currentThread().interrupt();
                break;

            } catch (Exception e) {

                System.out.println(
                    "Camera processing error: " +
                    e.getMessage()
                );

                e.printStackTrace();

                try {
                    Thread.sleep(100);
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }

        frame.release();
    }

    // ================ YOLO PREPROCESSING + DETECTION ================

    private List<Detection> detectVehicles(Mat frame)
        throws Exception {

        List<Detection> detections = new ArrayList<>();

        Mat resized = new Mat();
        Mat padded = new Mat();
        Mat rgb = new Mat();

        List<Mat> channels = new ArrayList<>();

        float scale = Math.min(
            INPUT_SIZE / (float) frame.cols(),
            INPUT_SIZE / (float) frame.rows()
        );

        int newWidth = Math.round(frame.cols() * scale);
        int newHeight = Math.round(frame.rows() * scale);

        int padX = (INPUT_SIZE - newWidth) / 2;
        int padY = (INPUT_SIZE - newHeight) / 2;

        try {

            // Resize while preserving aspect ratio

            Imgproc.resize(
                frame,
                resized,
                new Size(newWidth, newHeight)
            );

            // Letterbox padding

            Core.copyMakeBorder(
                resized,
                padded,
                padY,
                INPUT_SIZE - newHeight - padY,
                padX,
                INPUT_SIZE - newWidth - padX,
                Core.BORDER_CONSTANT,
                new Scalar(114, 114, 114)
            );

            // BGR to RGB

            Imgproc.cvtColor(
                padded,
                rgb,
                Imgproc.COLOR_BGR2RGB
            );

            // Normalize pixels from 0-255 to 0-1

            rgb.convertTo(
                rgb,
                CvType.CV_32F,
                1.0 / 255.0
            );

            Core.split(rgb, channels);

            int planeSize = INPUT_SIZE * INPUT_SIZE;

            float[] inputData = new float[3 * planeSize];

            for (int c = 0; c < 3; c++) {

                float[] channelData = new float[planeSize];

                channels.get(c).get(
                    0, 0, channelData
                );

                System.arraycopy(
                    channelData,
                    0,
                    inputData,
                    c * planeSize,
                    planeSize
                );
            }

            String inputName =
                session.getInputNames().iterator().next();

            try (
                OnnxTensor inputTensor =
                    OnnxTensor.createTensor(
                        environment,
                        FloatBuffer.wrap(inputData),
                        new long[]{
                            1, 3, INPUT_SIZE, INPUT_SIZE
                        }
                    )
            ) {

                Map<String, OnnxTensor> inputs = new HashMap<>();

                inputs.put(inputName, inputTensor);

                try (OrtSession.Result results = session.run(inputs)) {

                    OnnxValue outputValue = results.get(0);

                    if (!(outputValue instanceof OnnxTensor)) {

                        throw new Exception(
                            "YOLO output is not an ONNX tensor."
                        );
                    }

                    OnnxTensor output = (OnnxTensor) outputValue;

                    Object rawOutput = output.getValue();

                    if (!(rawOutput instanceof float[][][])) {

                        throw new Exception(
                            "Unexpected YOLO output data type: " +
                            rawOutput.getClass().getName()
                        );
                    }

                    float[][][] outputData =
                        (float[][][]) rawOutput;

                    long[] shape =
                        output.getInfo().getShape();

                    if (!outputInfoPrinted) {

                        System.out.println(
                            "YOLO OUTPUT SHAPE: " +
                            Arrays.toString(shape)
                        );

                        outputInfoPrinted = true;
                    }

                    if (shape.length != 3) {

                        throw new Exception(
                            "Expected YOLO output [1,84,8400] " +
                            "or [1,8400,84]. Got: " +
                            Arrays.toString(shape)
                        );
                    }

                    boolean channelsFirst =
                        shape[1] < shape[2];

                    int features = (int)(
                        channelsFirst ? shape[1] : shape[2]
                    );

                    int candidates = (int)(
                        channelsFirst ? shape[2] : shape[1]
                    );

                    // Standard YOLOv8 COCO output has 84 features:
                    // 4 box coordinates + 80 class scores.

                    if (features != 84) {

                        throw new Exception(
                            "This code expects standard YOLOv8n COCO " +
                            "output with 84 features.\n" +
                            "Your model output is: " +
                            Arrays.toString(shape)
                        );
                    }

                    float maxVehicleScore = 0;

                    for (int i = 0; i < candidates; i++) {

                        float centerX = getValue(
                            outputData,
                            channelsFirst,
                            i,
                            0
                        );

                        float centerY = getValue(
                            outputData,
                            channelsFirst,
                            i,
                            1
                        );

                        float width = getValue(
                            outputData,
                            channelsFirst,
                            i,
                            2
                        );

                        float height = getValue(
                            outputData,
                            channelsFirst,
                            i,
                            3
                        );

                        float bestConfidence = 0;
                        int bestClass = -1;

                        for (int j = 4; j < features; j++) {

                            int classId = j - 4;

                            if (!isVehicle(classId)) {
                                continue;
                            }

                            float score = getValue(
                                outputData,
                                channelsFirst,
                                i,
                                j
                            );

                            if (score > maxVehicleScore) {
                                maxVehicleScore = score;
                            }

                            if (score > bestConfidence) {

                                bestConfidence = score;
                                bestClass = classId;
                            }
                        }

                        if (
                            bestClass == -1 ||
                            bestConfidence < CONFIDENCE_THRESHOLD
                        ) {
                            continue;
                        }

                        // Convert YOLO letterbox coordinates
                        // back to original camera coordinates.

                        float x1 = (centerX - width / 2 - padX) / scale;
                        float y1 = (centerY - height / 2 - padY) / scale;

                        float x2 = (centerX + width / 2 - padX) / scale;
                        float y2 = (centerY + height / 2 - padY) / scale;

                        int left = Math.max(
                            0,
                            Math.min(frame.cols(), (int) x1)
                        );

                        int top = Math.max(
                            0,
                            Math.min(frame.rows(), (int) y1)
                        );

                        int right = Math.max(
                            0,
                            Math.min(frame.cols(), (int) x2)
                        );

                        int bottom = Math.max(
                            0,
                            Math.min(frame.rows(), (int) y2)
                        );

                        if (right <= left || bottom <= top) {
                            continue;
                        }

                        Rect box = new Rect(
                            left,
                            top,
                            right - left,
                            bottom - top
                        );

                        detections.add(
                            new Detection(
                                bestClass,
                                bestConfidence,
                                box
                            )
                        );
                    }

                    if (outputInfoPrinted) {

                        // Print maximum score once for debugging.
                        System.out.println(
                            "Highest vehicle confidence: " +
                            maxVehicleScore
                        );

                        // Prevent printing on every frame.
                        outputInfoPrinted = false;
                    }
                }
            }

            return applyNMS(detections);

        } finally {

            resized.release();
            padded.release();
            rgb.release();

            for (Mat channel : channels) {
                channel.release();
            }
        }
    }

    // ================ READ YOLO OUTPUT ================

    private float getValue(
        float[][][] output,
        boolean channelsFirst,
        int candidate,
        int feature
    ) {

        if (channelsFirst) {
            return output[0][feature][candidate];
        }

        return output[0][candidate][feature];
    }

    // ================ VEHICLE CLASS CHECK ================

    private boolean isVehicle(int classId) {

        for (int id : VEHICLE_IDS) {

            if (id == classId) {
                return true;
            }
        }

        return false;
    }

    private String getVehicleName(int classId) {

        for (int i = 0; i < VEHICLE_IDS.length; i++) {

            if (VEHICLE_IDS[i] == classId) {
                return VEHICLE_NAMES[i];
            }
        }

        return "VEHICLE";
    }

    // ================ NON MAXIMUM SUPPRESSION ================

    private List<Detection> applyNMS(
        List<Detection> detections
    ) {

        detections.sort(
            Comparator.comparingDouble(
                d -> -d.confidence
            )
        );

        List<Detection> result = new ArrayList<>();

        boolean[] removed = new boolean[detections.size()];

        for (int i = 0; i < detections.size(); i++) {

            if (removed[i]) {
                continue;
            }

            Detection current = detections.get(i);

            result.add(current);

            for (int j = i + 1; j < detections.size(); j++) {

                if (removed[j]) {
                    continue;
                }

                Detection next = detections.get(j);

                if (
                    current.classId == next.classId &&
                    calculateIoU(current.box, next.box) > NMS_THRESHOLD
                ) {
                    removed[j] = true;
                }
            }
        }

        return result;
    }

    private float calculateIoU(Rect a, Rect b) {

        int x1 = Math.max(a.x, b.x);
        int y1 = Math.max(a.y, b.y);

        int x2 = Math.min(
            a.x + a.width,
            b.x + b.width
        );

        int y2 = Math.min(
            a.y + a.height,
            b.y + b.height
        );

        int intersection =
            Math.max(0, x2 - x1) *
            Math.max(0, y2 - y1);

        float union =
            (float) a.width * a.height +
            (float) b.width * b.height -
            intersection;

        if (union <= 0) {
            return 0;
        }

        return intersection / union;
    }

    // ================ FOUR ZONES ================

    private int getZone(
        int x,
        int y,
        int width,
        int height
    ) {

        boolean right = x >= width / 2;
        boolean bottom = y >= height / 2;

        if (!right && !bottom) {
            return 0;
        }

        if (right && !bottom) {
            return 1;
        }

        if (right && bottom) {
            return 2;
        }

        return 3;
    }

    // ================ DRAW ZONE DIVISIONS ================

    private void drawZones(Mat frame) {

        int middleX = frame.cols() / 2;
        int middleY = frame.rows() / 2;

        Imgproc.line(
            frame,
            new Point(middleX, 0),
            new Point(middleX, frame.rows()),
            new Scalar(255, 255, 255),
            2
        );

        Imgproc.line(
            frame,
            new Point(0, middleY),
            new Point(frame.cols(), middleY),
            new Scalar(255, 255, 255),
            2
        );

        Point[] positions = {
            new Point(20, 35),
            new Point(middleX + 20, 35),
            new Point(middleX + 20, middleY + 35),
            new Point(20, middleY + 35)
        };

        for (int i = 0; i < 4; i++) {

            Imgproc.putText(
                frame,
                ZONE_NAMES[i],
                positions[i],
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.9,
                ZONE_COLORS[i],
                2
            );
        }
    }

    // ================ DRAW VEHICLE BOXES ================

    private void drawDetection(
        Mat frame,
        Detection detection,
        int zone
    ) {

        Imgproc.rectangle(
            frame,
            detection.box,
            ZONE_COLORS[zone],
            2
        );

        String label = getVehicleName(detection.classId);

        label += String.format(
            " %.0f%%",
            detection.confidence * 100
        );

        Imgproc.putText(
            frame,
            label,
            new Point(
                detection.box.x,
                Math.max(20, detection.box.y - 8)
            ),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.6,
            ZONE_COLORS[zone],
            2
        );
    }

    // ================ TRAFFIC SIGNAL ================

    private void updateSignal() {

        long elapsed =
            (System.currentTimeMillis() - signalStartTime) / 1000;

        if (elapsed >= SIGNAL_DURATION) {

            activeSignal = (activeSignal + 1) % 4;

            signalStartTime = System.currentTimeMillis();

            elapsed = 0;
        }

        remainingSeconds =
            SIGNAL_DURATION - (int) elapsed;

        signalLabel.setText(
            ZONE_NAMES[activeSignal] + " GREEN"
        );

        timerLabel.setText(
            remainingSeconds + " SECONDS"
        );

        signalLabel.setForeground(Color.GREEN);
    }

    private void drawSignalOnCamera(Mat frame) {

        Imgproc.putText(
            frame,
            "GREEN SIGNAL: " + ZONE_NAMES[activeSignal],
            new Point(20, frame.rows() - 25),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.8,
            new Scalar(0, 255, 0),
            2
        );
    }

    // ================ DASHBOARD COUNTS ================

    private void updateDashboard() {

        int total = 0;

        for (int count : zoneCounts) {
            total += count;
        }

        final int totalCount = total;
        final int[] snapshot = zoneCounts.clone();

        SwingUtilities.invokeLater(() -> {

            totalLabel.setText(
                "TOTAL VEHICLES: " + totalCount
            );

            for (int i = 0; i < 4; i++) {

                zoneLabels[i].setText(
                    ZONE_NAMES[i] + ": " + snapshot[i]
                );
            }
        });
    }

    // ================ DISPLAY FRAME ================

    private void displayFrame(Mat frame) {

        Mat display = new Mat();
        Mat bgr = new Mat();

        try {

            Imgproc.resize(
                frame,
                display,
                new Size(900, 550)
            );

            if (display.type() != CvType.CV_8UC3) {

                display.convertTo(
                    bgr,
                    CvType.CV_8UC3
                );

            } else {

                display.copyTo(bgr);
            }

            byte[] data = new byte[
                (int) (bgr.total() * bgr.channels())
            ];

            bgr.get(0, 0, data);

            BufferedImage image = new BufferedImage(
                bgr.cols(),
                bgr.rows(),
                BufferedImage.TYPE_3BYTE_BGR
            );

            image.getRaster().setDataElements(
                0,
                0,
                bgr.cols(),
                bgr.rows(),
                data
            );

            SwingUtilities.invokeLater(() -> {

                cameraLabel.setIcon(new ImageIcon(image));
                cameraLabel.setText("");
            });

        } finally {

            display.release();
            bgr.release();
        }
    }

    // ================ CLEANUP ================

    private void shutdown() {

        running = false;

        if (cameraThread != null) {
            cameraThread.interrupt();
        }

        if (camera != null) {
            camera.release();
        }

        try {

            if (session != null) {
                session.close();
            }

            if (environment != null) {
                environment.close();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        if (window != null) {
            window.dispose();
        }

        System.out.println("SMART TRAFFIC AI CLOSED.");
    }
}
