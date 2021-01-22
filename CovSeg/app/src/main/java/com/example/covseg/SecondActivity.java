package com.example.covseg;


import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.drawable.Drawable;



import android.util.Log;
import android.view.Menu;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import android.content.Intent;
import android.graphics.Bitmap;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.Tensor;
import org.pytorch.Module;
import org.pytorch.IValue;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SecondActivity extends AppCompatActivity {

    Button load;
    ImageView imageView;
    Button segment;
    TextView textview;
    TextView textview1;
    TextView textview2;
    TextView textview3;
    private static final int FILE_SELECT_CODE = 0;
    public Bitmap bitmap;
    /* private static final String TAG = null; */

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_secondary);

        load = findViewById(R.id.button2);
        segment = findViewById(R.id.button3);
        segment.setEnabled(false);
        imageView = findViewById(R.id.imageView2);
        textview  = findViewById(R.id.textView2);
        textview1  = findViewById(R.id.textView3);
        textview2 = findViewById(R.id.textView4);
        textview3 = findViewById(R.id.textView5);
        load.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (ContextCompat.checkSelfPermission(SecondActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                    selectimage();
                }
                else
                {
                    ActivityCompat.requestPermissions(SecondActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 2);
                }

            }

        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if(requestCode == 2 && grantResults[0]==PackageManager.PERMISSION_GRANTED)
        {
            selectimage();
        }
        else
        {
            Toast.makeText(SecondActivity.this,"Please Provide Permission", Toast.LENGTH_SHORT).show();
        }
    }

    private void selectimage() {

        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent, 86);


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        if (requestCode == 86 && resultCode == RESULT_OK && data!=null)
            try {
                // We need to recyle unused bitmaps
                if (bitmap != null) {
                    bitmap.recycle();
                }
                InputStream stream = getContentResolver().openInputStream(
                        data.getData());
                bitmap = BitmapFactory.decodeStream(stream);
                stream.close();
                imageView.setImageBitmap(bitmap);
                textview.setText("Input CT-Scan");
                textview1.setText("");
                textview2.setText("");
                textview3.setText("");
                segment.setEnabled(true);
                segment.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        openNewActivity();
                    }
                });
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        super.onActivityResult(requestCode, resultCode, data);


    }

    public void openNewActivity() {
        //Intent intent = new Intent(this, TeritiaryActivity.class);
        //intent.putExtra("BitmapImage", bitmap);
        //startActivity(intent);
        Module module = null;
        try {
            module = Module.load(assetFilePath(this, "model4.pt"));
        } catch (IOException e) {
            Log.e("COvSeg", "Error reading assets", e);

            finish();
        }

        float[] mean = {0.0f,0.0f,0.0f};
        float[] std = {1.0f,1.0f,1.0f};

        long tStart = System.currentTimeMillis();

        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, 0,0, 512,512, mean, std); // converts to [0,1] and then does (x-mu)/sigma


        //long [] hh = inputTensor.shape();

        // running the model

        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        long [] hh = outputTensor.shape();

        float[] array1 = outputTensor.getDataAsFloatArray() ;

        Bitmap OutBitmap = Bitmap.createBitmap(512 , 512, Bitmap.Config.RGB_565);
        int[] pixels = new int[512 * 512];
        int countA=0;
        int countB=0;
        for (int i = 0; i < 512 * 512; i++){
            if (array1[i]>1.0){
                pixels[i]= Color.argb(100, 0, 255, 0);
                countB++;}
            else if (array1[i]>0.0){
                pixels[i]= Color.argb(100, 255, 0, 0);
                countA++;}
            else{
                pixels[i]=Color.argb(100, 0, 0, 0);}}

        OutBitmap.setPixels(pixels, 0, 512, 0, 0, 512, 512);

        long tEnd = System.currentTimeMillis();
        long tDelta = tEnd - tStart;
        double elapsedSeconds = tDelta / 1000.0;

        DecimalFormat df = new DecimalFormat("#0.00");
        float A = (((float)countA)/(countA+countB))*100;
        float B = (((float)countB)/(countA+countB))*100;
        textview1.setText("Abnormal Region (Red): " + df.format(A) + " %");
        textview2.setText("Normal Region (Green): " + df.format(B) + " %");
        textview3.setText("Elapsed Time (Seconds): " + df.format(elapsedSeconds));

        textview.setText("Segmented COVID-19 Anomalies");
        imageView.setImageBitmap(OutBitmap);


    }


    public static String assetFilePath (Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}
