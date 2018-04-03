package com.example.samueltallent.androidcamera2api;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.view.View;
import android.graphics.Bitmap;

public class CanvasView extends View {

    public CanvasView(Context context) {
        super(context);
    }

    public CanvasView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    private Bitmap bitmap = null;

    public CanvasView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    protected void setBitmap(Bitmap map){
        bitmap = map;
    }
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if(bitmap != null) {
            Paint p = new Paint();
            canvas.drawBitmap(bitmap, 0, 0, p);
        }
    }
}
