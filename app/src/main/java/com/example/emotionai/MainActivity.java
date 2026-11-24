package com.example.emotionai;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.exifinterface.media.ExifInterface;

import com.example.emotionai.ai.EmotionClassifier;
import com.example.emotionai.detection.FaceDetectorHelper;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private FaceDetectorHelper faceDetectorHelper;
    private EmotionClassifier emotionClassifier;

    private ImageView imagePreview;
    private Button btnCamera, btnGallery, btnAnalyze;
    private ProgressBar progressBar;
    private Bitmap selectedBitmap;

    private ActivityResultLauncher<String> galleryLauncher;
    private ActivityResultLauncher<Void> cameraLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imagePreview = findViewById(R.id.imagePreview);
        btnCamera = findViewById(R.id.btnCamera);
        btnGallery = findViewById(R.id.btnGallery);
        btnAnalyze = findViewById(R.id.btnAnalyze);
        progressBar = findViewById(R.id.progressBar);

        faceDetectorHelper = new FaceDetectorHelper();

        try {
            emotionClassifier = new EmotionClassifier(this);
        } catch (IOException | RuntimeException e) {
            emotionClassifier = null;
            Toast.makeText(this,
                    "Could not load emotion model: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
        }

        progressBar.setVisibility(View.GONE);
        btnAnalyze.setEnabled(false);

        galleryLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
                        loadImage(uri);
                    }
                });

        cameraLauncher = registerForActivityResult(
                new ActivityResultContracts.TakePicturePreview(),
                bitmap -> {
                    if (bitmap != null) {
                        Bitmap softwareBitmap = bitmap;
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                            if (bitmap.getConfig() == Bitmap.Config.HARDWARE) {
                                softwareBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                            }
                        }
                        showSelectedImage(softwareBitmap);
                    }
                });

        btnGallery.setOnClickListener(v -> galleryLauncher.launch("image/*"));
        btnCamera.setOnClickListener(v -> cameraLauncher.launch(null));
        btnAnalyze.setOnClickListener(v -> analyzeImage());
    }

    private void showSelectedImage(Bitmap bitmap) {
        selectedBitmap = bitmap;
        imagePreview.setImageBitmap(bitmap);
        btnAnalyze.setEnabled(selectedBitmap != null && emotionClassifier != null);
    }

    private void loadImage(Uri uri) {
        try {
            Bitmap bitmap = loadBitmapFromUri(uri);
            if (bitmap != null) {
                showSelectedImage(bitmap);
            } else {
                Toast.makeText(this, "Unable to load image.", Toast.LENGTH_SHORT).show();
            }
        } catch (IOException | RuntimeException e) {
            Toast.makeText(this, "Unable to load image: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
        }
    }

    private Bitmap loadBitmapFromUri(Uri uri) throws IOException {
        InputStream input = getContentResolver().openInputStream(uri);
        Bitmap bitmap = BitmapFactory.decodeStream(input);
        if (input != null) {
            input.close();
        }
        if (bitmap == null) {
            return null;
        }

        // Handle EXIF orientation (rotates bitmap so faces are upright for ML Kit)
        int rotation = 0;
        try (InputStream exifInput = getContentResolver().openInputStream(uri)) {
            if (exifInput != null) {
                ExifInterface exif = new ExifInterface(exifInput);
                int orientation = exif.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL);
                switch (orientation) {
                    case ExifInterface.ORIENTATION_ROTATE_90:
                        rotation = 90;
                        break;
                    case ExifInterface.ORIENTATION_ROTATE_180:
                        rotation = 180;
                        break;
                    case ExifInterface.ORIENTATION_ROTATE_270:
                        rotation = 270;
                        break;
                }
            }
        } catch (Exception ignored) {
        }

        if (rotation != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotation);
            bitmap = Bitmap.createBitmap(
                    bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if (bitmap.getConfig() == Bitmap.Config.HARDWARE) {
                bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            }
        }

        return bitmap;
    }

    private void setAnalyzing(boolean analyzing) {
        progressBar.setVisibility(analyzing ? View.VISIBLE : View.GONE);
        btnAnalyze.setEnabled(!analyzing
                && selectedBitmap != null
                && emotionClassifier != null);
        btnCamera.setEnabled(!analyzing);
        btnGallery.setEnabled(!analyzing);
    }

    private void analyzeImage() {
        if (selectedBitmap == null) {
            Toast.makeText(this, "Please select an image first.",
                    Toast.LENGTH_SHORT).show();
            return;
        }
        if (emotionClassifier == null) {
            Toast.makeText(this, "Emotion model is not initialized.",
                    Toast.LENGTH_LONG).show();
            return;
        }

        setAnalyzing(true);

        faceDetectorHelper.detectFace(
                selectedBitmap,
                new FaceDetectorHelper.FaceCallback() {
                    @Override
                    public void onFaceDetected(Bitmap faceBitmap) {
                        runOnUiThread(() -> {
                            try {
                                EmotionClassifier.Prediction prediction =
                                        emotionClassifier.predictWithConfidence(faceBitmap);

                                @SuppressLint("ThreadConstraint") byte[] faceBytes = bitmapToBytes(faceBitmap);
                                if (faceBytes == null) {
                                    Toast.makeText(MainActivity.this,
                                            "Could not prepare face image.",
                                            Toast.LENGTH_LONG).show();
                                    return;
                                }

                                android.content.Intent intent =
                                        new android.content.Intent(
                                                MainActivity.this,
                                                ResultActivity.class);
                                intent.putExtra(ResultActivity.EXTRA_FACE_IMAGE, faceBytes);
                                intent.putExtra(ResultActivity.EXTRA_EMOTION,
                                        prediction.getLabel());
                                intent.putExtra(ResultActivity.EXTRA_CONFIDENCE,
                                        prediction.getConfidence());
                                startActivity(intent);
                            } catch (RuntimeException e) {
                                Toast.makeText(MainActivity.this,
                                        "AI prediction failed: " + e.getMessage(),
                                        Toast.LENGTH_LONG).show();
                            } finally {
                                setAnalyzing(false);
                            }
                        });
                    }

                    @Override
                    public void onNoFace() {
                        runOnUiThread(() -> {
                            setAnalyzing(false);
                            Toast.makeText(MainActivity.this,
                                    "No face detected. Try another image.",
                                    Toast.LENGTH_LONG).show();
                        });
                    }

                    @Override
                    public void onError(Exception e) {
                        runOnUiThread(() -> {
                            setAnalyzing(false);
                            Toast.makeText(MainActivity.this,
                                    "Face detection failed: " + e.getMessage(),
                                    Toast.LENGTH_LONG).show();
                        });
                    }
                });
    }

    private byte[] bitmapToBytes(Bitmap bitmap) {
        int maxSide = 512;
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float scale = Math.min(1.0f, (float) maxSide / Math.max(width, height));
        Bitmap smallBitmap = bitmap;
        if (scale < 1.0f) {
            smallBitmap = Bitmap.createScaledBitmap(
                    bitmap, Math.max(1, Math.round(width * scale)),
                    Math.max(1, Math.round(height * scale)), true);
        }

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        boolean success = smallBitmap.compress(Bitmap.CompressFormat.JPEG, 88, output);
        return success ? output.toByteArray() : null;
    }

    @Override
    protected void onDestroy() {
        if (faceDetectorHelper != null) {
            faceDetectorHelper.close();
        }
        if (emotionClassifier != null) {
            emotionClassifier.close();
        }
        super.onDestroy();
    }
}
