package com.example.emotionai.ai;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class EmotionClassifier {

    private static final String MODEL_FILE = "emotion_model.tflite";

    // Keep this label order identical to the order used to train your model.
    private final String[] labels = {
            "Angry", "Disgust", "Fear", "Happy",
            "Sad", "Surprise", "Neutral"
    };

    private Interpreter interpreter;
    private int inputWidth = 48;
    private int inputHeight = 48;
    private int inputChannels = 1;
    private boolean isFloatInput = true;




    public void inspectModel() {

        int[] inputShape =
                interpreter.getInputTensor(0).shape();

        int[] outputShape =
                interpreter.getOutputTensor(0).shape();

        String inputType =
                interpreter.getInputTensor(0).dataType().toString();

        String outputType =
                interpreter.getOutputTensor(0).dataType().toString();

        android.util.Log.d("EmotionAI",
                "Input Shape: " +
                        java.util.Arrays.toString(inputShape));

        android.util.Log.d("EmotionAI",
                "Input Type: " + inputType);

        android.util.Log.d("EmotionAI",
                "Output Shape: " +
                        java.util.Arrays.toString(outputShape));

        android.util.Log.d("EmotionAI",
                "Output Type: " + outputType);
    }

    public static class Prediction {
        private final String label;
        private final float confidence;

        public Prediction(String label, float confidence) {
            this.label = label;
            this.confidence = confidence;
        }

        public String getLabel() {
            return label;
        }

        public float getConfidence() {
            return confidence;
        }
    }


    public MappedByteBuffer loadModel(Context context) throws IOException {
        try (AssetFileDescriptor descriptor =
                     context.getAssets().openFd(MODEL_FILE);
             FileInputStream inputStream =
                     new FileInputStream(descriptor.getFileDescriptor());
             FileChannel channel = inputStream.getChannel()) {

            return channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    descriptor.getStartOffset(),
                    descriptor.getDeclaredLength());
        }
    }

    public EmotionClassifier(Context context) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(2);
        interpreter = new Interpreter(loadModel(context), options);


        try {
            Tensor inputTensor = interpreter.getInputTensor(0);
            int[] shape = inputTensor.shape();
            if (shape != null) {
                if (shape.length == 4) {
                    inputHeight = shape[1];
                    inputWidth = shape[2];
                    inputChannels = shape[3];
                } else if (shape.length == 3) {
                    inputHeight = shape[0];
                    inputWidth = shape[1];
                    inputChannels = shape[2];
                }
            }
            isFloatInput = (inputTensor.dataType() == DataType.FLOAT32);
        } catch (Exception ignored) {
        }
    }

    public synchronized String predict(Bitmap face) {
        return predictWithConfidence(face).getLabel();
    }

    public synchronized Prediction predictWithConfidence(Bitmap face) {
        if (face == null) {
            throw new IllegalArgumentException("Face bitmap is null.");
        }
        if (interpreter == null) {
            throw new IllegalStateException("Emotion model is closed.");
        }

        Bitmap resized = Bitmap.createScaledBitmap(face, inputWidth, inputHeight, true);

        int bytesPerNum = isFloatInput ? 4 : 1;
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(inputHeight * inputWidth * inputChannels * bytesPerNum);
        inputBuffer.order(ByteOrder.nativeOrder());

        for (int y = 0; y < inputHeight; y++) {
            for (int x = 0; x < inputWidth; x++) {
                int pixel = resized.getPixel(x, y);
                int red = (pixel >> 16) & 0xFF;
                int green = (pixel >> 8) & 0xFF;
                int blue = pixel & 0xFF;

                if (inputChannels == 1) {
                    float gray = (0.299f * red + 0.587f * green + 0.114f * blue) / 255.0f;
                    if (isFloatInput) {
                        inputBuffer.putFloat(gray);
                    } else {
                        inputBuffer.put((byte) (Math.round(gray * 255.0f) & 0xFF));
                    }
                } else {
                    if (isFloatInput) {
                        inputBuffer.putFloat(red / 255.0f);
                        inputBuffer.putFloat(green / 255.0f);
                        inputBuffer.putFloat(blue / 255.0f);
                    } else {
                        inputBuffer.put((byte) red);
                        inputBuffer.put((byte) green);
                        inputBuffer.put((byte) blue);
                    }
                }
            }
        }
        inputBuffer.rewind();

        int numClasses = labels.length;
        try {
            Tensor outputTensor = interpreter.getOutputTensor(0);
            int[] outShape = outputTensor.shape();
            if (outShape != null && outShape.length > 0) {
                numClasses = outShape[outShape.length - 1];
            }
        } catch (Exception ignored) {
        }

        float[][] output = new float[1][numClasses];
        interpreter.run(inputBuffer, output);

        int bestIndex = 0;
        int maxIndex = Math.min(numClasses, labels.length);
        for (int i = 1; i < maxIndex; i++) {
            if (output[0][i] > output[0][bestIndex]) {
                bestIndex = i;
            }
        }

        float confidence = calculateConfidence(output[0], bestIndex);
        String label = (bestIndex >= 0 && bestIndex < labels.length) ? labels[bestIndex] : "Unknown";
        return new Prediction(label, confidence);
    }

    private float calculateConfidence(float[] scores, int bestIndex) {
        if (scores == null || scores.length == 0 || bestIndex < 0 || bestIndex >= scores.length) {
            return 0.0f;
        }

        float sum = 0.0f;
        boolean validProbabilities = true;
        for (float score : scores) {
            if (Float.isNaN(score) || Float.isInfinite(score) || score < 0.0f || score > 1.0f) {
                validProbabilities = false;
                break;
            }
            sum += score;
        }

        if (validProbabilities && sum > 0.0f && Math.abs(sum - 1.0f) < 0.05f) {
            return Math.max(0.0f, Math.min(1.0f, scores[bestIndex]));
        }

        float max = scores[0];
        for (float score : scores) {
            if (score > max) max = score;
        }

        double expSum = 0.0;
        double bestExp = Math.exp(scores[bestIndex] - max);
        for (float score : scores) {
            expSum += Math.exp(score - max);
        }

        if (expSum <= 0.0 || Double.isNaN(expSum) || Double.isInfinite(expSum)) {
            return 0.0f;
        }
        return (float) (bestExp / expSum);
    }

    public synchronized void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }

}
