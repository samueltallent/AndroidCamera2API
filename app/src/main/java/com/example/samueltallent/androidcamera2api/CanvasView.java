package com.example.samueltallent.androidcamera2api;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.view.View;
import android.graphics.Bitmap;
import java.lang.Math;

import org.opencv.core.Point;

public class CanvasView extends View {

    public CanvasView(Context context) {
        super(context);
    }

    public CanvasView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    private Bitmap bitmap = null;
    private double angle = 0;

    public CanvasView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    protected void setBitmap(Bitmap map, double angel){
        bitmap = map;
        angle = angel;
    }
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if(bitmap != null) {
            Paint p = new Paint();
            p.setStrokeWidth(5);
            p.setColor(Color.BLUE);
            canvas.drawBitmap(bitmap, 0, 0, p);
            double width = canvas.getWidth();
            double height = canvas.getHeight();
            canvas.drawLine((float)width, (float)(height / 2), (float)(width - 400 * Math.cos(angle)),(float)(height/2 + 400 * Math.sin(angle)), p);
        }
    }
}
