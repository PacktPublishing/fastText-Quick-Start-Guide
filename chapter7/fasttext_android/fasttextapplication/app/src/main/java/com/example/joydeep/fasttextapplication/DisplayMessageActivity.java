package com.example.joydeep.fasttextapplication;

import android.content.Intent;
import android.content.res.AssetManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;

import fasttext.FastText;
import fasttext.FastTextPrediction;

public class DisplayMessageActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_display_message);

        // Get the Intent that started this activity and extract the string
        Intent intent = getIntent();
        String message = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);

        // Get the assets from the asset manager.
        AssetManager assetManager = getAssets();
        InputStream inputStream = null;
        try {
            inputStream = assetManager.open("dbpedia.ftz");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // load the model and make the predictions
        FastTextPrediction label = null;
        try {
            FastText model = FastText.loadModel(inputStream);
            label = model.predict(message);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Capture the layout's TextView and set the string as its text
        TextView textView = findViewById(R.id.textView);

        // show the predicted value
        textView.setText(label.label());

    }
}
