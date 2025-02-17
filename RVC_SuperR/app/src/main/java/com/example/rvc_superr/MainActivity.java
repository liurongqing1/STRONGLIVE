package com.example.rvc_superr;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.hardware.Camera.Parameters;
import android.hardware.Camera.PreviewCallback;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import android.Manifest;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.IOException;
import java.util.concurrent.ArrayBlockingQueue;

public class MainActivity extends Activity  implements SurfaceHolder.Callback,PreviewCallback{

    private SurfaceView surfaceview;

    private SurfaceHolder surfaceHolder;

    private Camera camera;

    private Parameters parameters;
    private Client mClient;
    private Button startLiveButton;
    private Button endLiveButton;

    int width = 640;//960;//1280;//640;//1280;//1280;
    int height = 360;//540;//360;//720;2160

    int framerate = 30;

    int biterate = 8500*1000;

    private static int yuvqueuesize = 10;

    //待解码视频缓冲队列，静态成员！
    public static ArrayBlockingQueue<byte[]> YUVQueue = new ArrayBlockingQueue<byte[]>(yuvqueuesize);

    private AvcEncoder avcCodec;
    // 权限请求的识别码
    // 权限请求的识别码
    private static final int REQUEST_CAMERA_PERMISSION = 200;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 初始化按钮和其他视图
        surfaceview = (SurfaceView)findViewById(R.id.surfaceview);
        startLiveButton = (Button)findViewById(R.id.begin);
        endLiveButton = (Button)findViewById(R.id.end);
        surfaceHolder = surfaceview.getHolder();
        surfaceHolder.addCallback(this);

        // 设置按钮的点击监听器
        startLiveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startLiveStreaming();
            }
        });

        endLiveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                stopLiveStreaming();
            }
        });

    }

    private void startLiveStreaming() {
        // 检查摄像头权限是否已被授予
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            // 如果没有授予权限，请求摄像头权限
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    REQUEST_CAMERA_PERMISSION);
        } else {
            // 如果权限已被授予，继续执行直播操作
            initializeLiveStreaming();
        }
    }

    // 当获取到权限结果时回调
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // 权限被用户同意，可以执行操作
                initializeLiveStreaming();
            } else {
                // 权限被用户拒绝，通过Toast或其他方式告知用户
                Toast.makeText(this, "需要摄像头权限来进行直播", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // 摄像头权限被授予后才调用这个方法
    private void initializeLiveStreaming() {
        // 创建客户端实例并连接服务器
        mClient = Client.getInstance();
        mClient.connect();

        // 摄像头操作，确保已被正确初始化
        if (camera == null) {
            camera = getBackCamera();
            startcamera(camera);
        }

        // 创建编码器并启动编码线程
        if (avcCodec == null) {
            avcCodec = new AvcEncoder(width, height, framerate, biterate, mClient);
            avcCodec.StartEncoderThread(); // 启动编码线程
        }
    }

    private void stopLiveStreaming() {
        if(avcCodec != null) {
            avcCodec.StopThread(); // 停止编码线程
            avcCodec = null;
        }
        if(camera != null) {
            camera.stopPreview();
            camera.setPreviewCallback(null);
            camera.release(); // 关闭摄像头
            camera = null;
        }
        // 断开网络连接
        if(mClient != null) {
            mClient.disconnect();
            mClient = null;
        }
    }

    // 放在onDestroy方法中的逻辑可以删除，因为停止直播的按钮将处理这些资源的释放。
// 但通常在onPause或onDestroy方法中添加释放资源的代码是一个很好的实践，以防用户直接退出应用。
    @Override
    protected void onDestroy() {
        stopLiveStreaming(); // 确保资源被释放
        super.onDestroy();
    }


    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        camera = getBackCamera();
        startcamera(camera);
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {

    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        // 确保资源被正确地清理
        stopLiveStreaming();
    }


    @Override
    public void onPreviewFrame(byte[] data, android.hardware.Camera camera) {
        //将当前帧图像保存在队列中
        putYUVData(data,data.length);
    }

    public void putYUVData(byte[] buffer, int length) {
        if (YUVQueue.size() >= 10) {
            YUVQueue.poll();
        }
        YUVQueue.add(buffer);
    }


    private void startcamera(Camera mCamera){
        if(mCamera != null){
            try {
                mCamera.setPreviewCallback(this);
                mCamera.setDisplayOrientation(90);
                if(parameters == null){
                    parameters = mCamera.getParameters();
                }
                //获取默认的camera配置
                parameters = mCamera.getParameters();
                //设置预览格式
                parameters.setPreviewFormat(ImageFormat.NV21);
                //设置预览图像分辨率
                parameters.setPreviewSize(width, height);
                //配置camera参数
                mCamera.setParameters(parameters);
                //将完全初始化的SurfaceHolder传入到setPreviewDisplay(SurfaceHolder)中
                //没有surface的话，相机不会开启preview预览
                mCamera.setPreviewDisplay(surfaceHolder);
                //调用startPreview()用以更新preview的surface，必须要在拍照之前start Preview
                mCamera.startPreview();

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private Camera getBackCamera() {
        Camera c = null;
        try {
            //获取Camera的实例
            c = Camera.open(0);
        } catch (Exception e) {
            e.printStackTrace();
        }
        //获取Camera的实例失败时返回null
        return c;
    }


}