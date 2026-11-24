
package com.example.emotionai.detection;

import android.graphics.Bitmap;
import android.graphics.Rect;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.util.List;

public class FaceDetectorHelper {

    private final FaceDetector detector;

    public FaceDetectorHelper() {
        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(
                                FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE
                        )
                        .setMinFaceSize(0.15f)
                        .build();

        detector = FaceDetection.getClient(options);
    }

    public interface FaceCallback {
        void onFaceDetected(Bitmap faceBitmap);
        void onNoFace();
        void onError(Exception e);
    }

    public void detectFace(
            Bitmap originalBitmap,
            FaceCallback callback
    ) {
        if (originalBitmap == null) {
            callback.onNoFace();
            return;
        }

        // ML Kit requires a software bitmap (ARGB_8888 or RGB_565). HARDWARE config causes an exception.
        Bitmap bitmapToProcess = originalBitmap;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            if (bitmapToProcess.getConfig() == Bitmap.Config.HARDWARE) {
                bitmapToProcess = bitmapToProcess.copy(Bitmap.Config.ARGB_8888, false);
            }
        }

        InputImage image = InputImage.fromBitmap(
                bitmapToProcess, 0
        );

        Bitmap finalBitmapToProcess = bitmapToProcess;
        detector.process(image)
                .addOnSuccessListener(faces -> {
                    if (faces == null || faces.isEmpty()) {
                        callback.onNoFace();
                        return;
                    }

                    // Select the largest detected face
                    Face largestFace = faces.get(0);
                    for (Face face : faces) {
                        if (face.getBoundingBox().width() * face.getBoundingBox().height()
                                > largestFace.getBoundingBox().width() * largestFace.getBoundingBox().height()) {
                            largestFace = face;
                        }
                    }

                    Rect bounds = largestFace.getBoundingBox();

                    // Add 15% margin padding around the detected face for optimal emotion model context
                    int marginX = (int) (bounds.width() * 0.15f);
                    int marginY = (int) (bounds.height() * 0.15f);

                    int width = finalBitmapToProcess.getWidth();
                    int height = finalBitmapToProcess.getHeight();

                    int left = Math.max(0, bounds.left - marginX);
                    int top = Math.max(0, bounds.top - marginY);
                    int right = Math.min(width, bounds.right + marginX);
                    int bottom = Math.min(height, bounds.bottom + marginY);

                    if (right <= left || bottom <= top) {
                        callback.onNoFace();
                        return;
                    }

                    Bitmap croppedFace = Bitmap.createBitmap(
                            finalBitmapToProcess,
                            left,
                            top,
                            right - left,
                            bottom - top
                    );

                    callback.onFaceDetected(croppedFace);

                })
                .addOnFailureListener(callback::onError);
    }

    public void close() {
        detector.close();
    }
}
