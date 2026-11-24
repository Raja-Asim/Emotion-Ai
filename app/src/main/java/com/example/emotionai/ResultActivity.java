package com.example.emotionai;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.Locale;

public class ResultActivity extends AppCompatActivity {

    public static final String EXTRA_FACE_IMAGE = "extra_face_image";
    public static final String EXTRA_EMOTION = "extra_emotion";
    public static final String EXTRA_CONFIDENCE = "extra_confidence";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        ImageView resultImage = findViewById(R.id.resultImage);
        TextView emotionText = findViewById(R.id.emotionText);
        TextView confidenceText = findViewById(R.id.confidenceText);
        Button btnAnalyzeAgain = findViewById(R.id.btnAnalyzeAgain);

        byte[] imageBytes = getIntent().getByteArrayExtra(EXTRA_FACE_IMAGE);
        String emotion = getIntent().getStringExtra(EXTRA_EMOTION);
        float confidence = getIntent().getFloatExtra(EXTRA_CONFIDENCE, -1.0f);

        if (imageBytes != null) {
            Bitmap bitmap = BitmapFactory.decodeByteArray(
                    imageBytes, 0, imageBytes.length);
            if (bitmap != null) {
                resultImage.setImageBitmap(bitmap);
            }
        }

        emotionText.setText(emotion == null || emotion.trim().isEmpty()
                ? "Emotion not available"
                : emotion);

        if (confidence >= 0.0f && confidence <= 1.0f) {
            confidenceText.setText(String.format(
                    Locale.getDefault(), "Confidence: %.1f%%", confidence * 100.0f));
        } else {
            confidenceText.setText("Confidence: unavailable");
        }

        btnAnalyzeAgain.setOnClickListener(v -> finish());
    }
}
