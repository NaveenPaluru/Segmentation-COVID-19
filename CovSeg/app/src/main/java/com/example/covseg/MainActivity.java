package com.example.covseg;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;



public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        WebView view = (WebView) findViewById(R.id.web1);
        WebSettings webSetting = view.getSettings();
        webSetting.setBuiltInZoomControls(true);
        webSetting.setJavaScriptEnabled(true);
        view.setWebViewClient(new WebViewClient());
        view.loadUrl("file:///android_asset/licence.html");


        final CheckBox checkBox = (CheckBox) findViewById(R.id.checkBox1);
        final Button b = (Button) findViewById(R.id.button1);

        checkBox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView,
                                         boolean isChecked) {
                if (buttonView.isChecked()) {
                    b.setEnabled(true);
                    b.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View v) {
                            openNewActivity();
                        }
                    });
                } else {
                    b.setEnabled(false);
                }
            }
        });


    }


    public void openNewActivity() {
        Intent intent = new Intent(this, SecondActivity.class);
        startActivity(intent);
    }
}




