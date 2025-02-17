package com.example.rvc_superr;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.arthenica.mobileffmpeg.Config;
import com.arthenica.mobileffmpeg.ExecuteCallback;
import com.arthenica.mobileffmpeg.FFmpeg;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CountDownLatch;


public class img_Activity extends AppCompatActivity {
    private static final int TIMEOUT_US = 10000;
    private static final int BIT_RATE = 4500 * 1000;
    private static final int FRAME_RATE = 30;
    private static final int I_FRAME_INTERVAL = 1;
    private static final String OUTPUT_FILE_NAME = "output_video.h264";
    private Button connectButton;
    private Button disconnectButton;
    private Button convertButton;
    private Client mClient;
    public byte[] configbyte;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_img);
        // 申请读写权限
        if (!hasPermission()) {
            requestPermission();
        }

        connectButton = findViewById(R.id.connectButton);  // 连接按钮
        disconnectButton = findViewById(R.id.disconnectButton);  // 断开连接按钮
        convertButton = findViewById(R.id.convertButton);  // 执行转换按钮

        mClient=Client.getInstance();


        // 连接按钮点击事件
        connectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mClient.connect();
            }
        });

        // 断开连接按钮点击事件
        disconnectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mClient.disconnect();
            }
        });

        // 执行转换按钮点击事件
        convertButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startConversion();  // 开始转换
            }
        });


    }

    private void startConversion() {
        new ConvertVideoTask().execute();
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                Toast.makeText(
                                img_Activity.this,
                                "Write external storage permission is required for this demo",
                                Toast.LENGTH_LONG)
                        .show();
            }
            requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }
    }


    private class ConvertVideoTask extends AsyncTask<Void, Void, Void> {
        boolean success = true;
        protected Void doInBackground(Void... voids) {
            // 在后台线程中提取帧
            new Thread(new Runnable() {
                @Override
                public void run() {
                    // 提取视频帧到文件夹
                    File[] frameFiles = new File[0];
                    try {
                        frameFiles = loadImagesWithFFmpeg_3();  // 提取视频帧
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    List<Bitmap> images = loadImages_2(frameFiles);  // 转换为 Bitmap

                    // 提取帧成功后，启动编码任务
                    if (images != null && !images.isEmpty()) {
                        // 使用 Handler 将编码任务发送到主线程执行
                        new Handler(Looper.getMainLooper()).post(new Runnable() {
                            @Override
                            public void run() {
                                try {
                                    Uri outputUri = createOutputUri();
                                    ParcelFileDescriptor pfd = getContentResolver().openFileDescriptor(outputUri, "w");
                                    FileOutputStream outputStream = new FileOutputStream(pfd.getFileDescriptor());
                                    //encodeImagesToH264(images, outputStream);  // 对图像进行编码并保存
                                    encodeImagesToH264AndSendToServer_2(images);
                                } catch (IOException e) {
                                    e.printStackTrace();
                                    success = false;
                                }
                            }
                        });
                    }
                }
            }).start();

            return null;
        }

        private File[] loadImagesWithFFmpeg_3() throws IOException {
            final File outputDir = new File(getExternalFilesDir(null), "extracted_frames");  // 输出目录
            // 确保输出目录存在
            if (!outputDir.exists()) {
                outputDir.mkdirs();
            }
            final String videoFileName = "Jockey_000_540.mp4";
            String videoFilePath = copyVideoFromAssets(videoFileName); // 拷贝视频文件到缓存目录

            long startTime = System.currentTimeMillis();  // 记录开始时间

            // FFmpeg 命令：提取视频帧为图片文件
            String command = "-i " + videoFilePath + " -vf fps=30 " + outputDir.getAbsolutePath() + "/frame_%04d.png";
            // 执行 FFmpeg 命令
            final CountDownLatch latch = new CountDownLatch(1);  // 用来同步等待FFmpeg执行完

            FFmpeg.executeAsync(command, new ExecuteCallback() {
                @Override
                public void apply(long executionId, int returnCode) {
                    if (returnCode == Config.RETURN_CODE_SUCCESS) {
                        Log.d("FFmpeg", "视频帧提取成功");
                    } else {
                        Log.e("FFmpeg", "视频帧提取失败，错误码：" + returnCode);
                    }
                    latch.countDown();  // 告知FFmpeg命令已执行完
                }
            });
            // 等待 FFmpeg 执行完毕（如果需要同步执行）
            while (!outputDir.exists() || outputDir.listFiles() == null || outputDir.listFiles().length == 0) {
                try {
                    Thread.sleep(100);  // 等待文件提取完成
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            long endTime = System.currentTimeMillis();  // 记录结束时间
            Log.d("test-time", "从MP4到图片转换总时间: " + (endTime - startTime) + "毫秒");

            return outputDir.listFiles();  // 返回提取的帧文件列表
        }

        private List<Bitmap> loadImages_2(File[] frameFiles) {
            final List<Bitmap> images = new ArrayList<>();
            // 逐帧加载图片并转换为 Bitmap
            if (frameFiles != null && frameFiles.length > 0) {
                // 按文件名中的数字部分排序，确保按顺序加载
                Arrays.sort(frameFiles, new Comparator<File>() {
                    @Override
                    public int compare(File f1, File f2) {
                        // 提取文件名部分 "frame_XXXX.png" 中的数字
                        String name1 = f1.getName();
                        String name2 = f2.getName();

                        // 假设文件名格式是 frame_XXXX.png, 提取数字部分
                        int num1 = Integer.parseInt(name1.replaceAll("\\D+", "")); // 提取数字
                        int num2 = Integer.parseInt(name2.replaceAll("\\D+", "")); // 提取数字

                        return Integer.compare(num1, num2);  // 按数字进行排序
                    }
                });

                long startTime = System.currentTimeMillis();  // 记录开始时间

                // 加载每一帧
                for (File frameFile : frameFiles) {
                    Log.d("FFmpeg", "帧文件路径: " + frameFile.getAbsolutePath());
                    Bitmap bitmap = BitmapFactory.decodeFile(frameFile.getAbsolutePath());
                    if (bitmap != null) {
                        images.add(bitmap);
                        // 打印加载的图片信息，包括文件名和内存大小
                        int imageSize = bitmap.getByteCount();  // 获取图片的内存大小（单位字节）
                        Log.d("FFmpeg", "加载帧: " + frameFile.getName() + ", 图片大小: " + imageSize + " 字节");
                    } else {
                        Log.e("FFmpeg", "无法加载帧: " + frameFile.getName());
                    }
                }

                long endTime = System.currentTimeMillis();  // 记录结束时间
                Log.d("test-time", "加载图片总时间: " + (endTime - startTime) + "毫秒");

            } else {
                Log.e("FFmpeg", "没有提取到任何帧文件！");
            }

            return images;
        }

        private String copyVideoFromAssets(String assetFileName) throws IOException {
            File outFile = new File(getCacheDir(), assetFileName);  // 缓存目录中的目标文件
            if (outFile.exists()) return outFile.getAbsolutePath();  // 如果文件已存在，直接返回路径

            InputStream inputStream = getAssets().open(assetFileName);
            FileOutputStream outputStream = new FileOutputStream(outFile);

            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, length);
            }

            outputStream.flush();
            outputStream.close();
            inputStream.close();

            return outFile.getAbsolutePath();  // 返回拷贝后的视频文件路径
        }

        private Uri createOutputUri() throws IOException {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                ContentResolver resolver = getContentResolver();
                ContentValues values = new ContentValues();
                values.put(MediaStore.MediaColumns.DISPLAY_NAME, OUTPUT_FILE_NAME);
                values.put(MediaStore.MediaColumns.MIME_TYPE, "video/avc");
                //values.put(MediaStore.MediaColumns.MIME_TYPE, "video/hevc");
                values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);
                return resolver.insert(MediaStore.Files.getContentUri("external"), values);
            } else {
                // 在Android Q以下，可以使用File来操作文件
                File directory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
                if (!directory.exists()) directory.mkdirs();
                File file = new File(directory, OUTPUT_FILE_NAME);
                return Uri.fromFile(file);
            }
        }

        @Override
        protected void onPostExecute(Void result) {
            if (success) {
                showToast("视频文件转换完成！");
            } else {
                showToast("视频文件转换失败，请稍后重试。");
            }
        }

        private void showToast(String message) {
            Toast.makeText(img_Activity.this, message, Toast.LENGTH_LONG).show();
        }

        private void encodeImagesToH264AndSendToServer_2(List<Bitmap> images) throws IOException {
            new Thread(new Runnable() {
                @Override
                public void run() {
                    // 使用MediaCodec创建编码器
                    MediaCodec codec = null;
                    byte[] configbyte = null; // 用于存储SPS/PPS配置数据

                    long totalEncodingTime = 0;  // 用来记录总的编码时间

                    try {
                        codec = MediaCodec.createEncoderByType("video/avc");
                        MediaFormat format = MediaFormat.createVideoFormat("video/avc", 960, 540);
                        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
                        format.setInteger(MediaFormat.KEY_BIT_RATE, BIT_RATE);
                        format.setInteger(MediaFormat.KEY_FRAME_RATE, FRAME_RATE);
                        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL);
                        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
                        codec.start();

                        // 获取输入和输出缓冲区
                        ByteBuffer[] inputBuffers = codec.getInputBuffers();
                        ByteBuffer[] outputBuffers = codec.getOutputBuffers();
                        MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();

                        long pts = 0;
                        long frameTime = 1000000 / FRAME_RATE;  // 计算帧时间

                        // 遍历图片列表并进行编码
                        for (Bitmap bitmap : images) {

                            long frameStartTime = System.currentTimeMillis();  // 记录每一帧编码开始时间

                            byte[] yuvData = getYUV420FromBitmap(bitmap);
                            if (yuvData == null || yuvData.length == 0) {
                                Log.e("FFmpeg-1", "YUV data is empty or null for bitmap: " + bitmap);
                                continue;
                            }

                            // 获取输入缓冲区并填充数据
                            int inputBufferIndex = codec.dequeueInputBuffer(TIMEOUT_US);
                            if (inputBufferIndex >= 0) {
                                ByteBuffer inputBuffer = inputBuffers[inputBufferIndex];
                                inputBuffer.clear();
                                inputBuffer.put(yuvData);
                                codec.queueInputBuffer(inputBufferIndex, 0, yuvData.length, pts, 0);
                                pts += frameTime;
                            }

                            // 获取输出缓冲区并处理编码后的帧数据
                            int outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US);

                            long frameEndTime = System.currentTimeMillis();  // 记录每一帧编码结束时间
                            long frameTimeTaken = frameEndTime - frameStartTime;
                            totalEncodingTime += frameTimeTaken;  // 累加帧编码时间
                            Log.d("test-time", "第 " + images.indexOf(bitmap) + " 帧编码时间: " + frameTimeTaken + " 毫秒");

                            while (outputBufferIndex >= 0) {
                                ByteBuffer outputBuffer = outputBuffers[outputBufferIndex];
                                byte[] outData = new byte[bufferInfo.size];
                                outputBuffer.get(outData);
                                Log.d("FFmpeg", "编码帧数据大小: " + bufferInfo.size + " 字节");

                                // 处理不同类型的帧
                                if (bufferInfo.flags == MediaCodec.BUFFER_FLAG_CODEC_CONFIG) {
                                    // 这是SPS/PPS数据
                                    configbyte = new byte[bufferInfo.size];
                                    System.arraycopy(outData, 0, configbyte, 0, bufferInfo.size);
                                    Log.d("SPS/PPS", "存储SPS/PPS数据");
                                } else if (bufferInfo.size > 0) {
                                    // 非配置帧，判断是否为关键帧
                                    byte[] keyframe = new byte[bufferInfo.size + (configbyte != null ? configbyte.length : 0)];

                                    if (configbyte != null) {
                                        // 如果有SPS/PPS数据，合并SPS/PPS和帧数据
                                        System.arraycopy(configbyte, 0, keyframe, 0, configbyte.length); // 将SPS/PPS填充到关键帧前面
                                        System.arraycopy(outData, 0, keyframe, configbyte.length, outData.length); // 将编码后的帧数据放在后面
                                        mClient.sendFrame(keyframe);  // 发送带有SPS/PPS的关键帧
                                    } else {
                                        // 发送普通帧
                                        mClient.sendFrame(outData);
                                    }
                                }

                                codec.releaseOutputBuffer(outputBufferIndex, false);
                                outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US);
                            }
                        }



                        // 停止并释放编码器
                        codec.stop();
                        codec.release();

                        Log.d("test-time", "从Bitmap到H264编码总时间: " + totalEncodingTime + " 毫秒");

                    } catch (IOException e) {
                        Log.e("EncodingError", "Error while encoding video", e);
                    } finally {
                        // 确保编码器被释放
                        if (codec != null) {
                            codec.stop();
                            codec.release();
                        }
                    }
                }
            }).start(); // 启动新线程
        }

        public byte[] getYUV420FromBitmap(Bitmap bitmap) {
            if (bitmap == null) {
                return null;
            }

            int inputWidth = bitmap.getWidth();
            int inputHeight = bitmap.getHeight();

            // 创建一个存储ARGB像素数据的数组
            int[] argb = new int[inputWidth * inputHeight];

            // 获取Bitmap对象中的像素数组
            bitmap.getPixels(argb, 0, inputWidth, 0, 0, inputWidth, inputHeight);

            // 创建YUV420SP格式的数组（NV21格式）
            byte[] yuv = new byte[inputWidth * inputHeight * 3 / 2];

            // 将ARGB数组转换为YUV420SP数组
            encodeYUV420SP(yuv, argb, inputWidth, inputHeight);

            // 回收Bitmap对象以节省内存
            bitmap.recycle();

            // 返回YUV420SP数组
            return yuv;
        }

        void encodeYUV420SP(byte[] yuv420sp, int[] argb, int width, int height) {
            final int frameSize = width * height;

            int yIndex = 0;
            int uvIndex = frameSize;

            int a, R, G, B, Y, U, V;
            int index = 0;
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    a = (argb[index] & 0xff000000) >> 24; // Alpha channel is not used
                    R = (argb[index] & 0xff0000) >> 16;
                    G = (argb[index] & 0xff00) >> 8;
                    B = (argb[index] & 0xff);
                    Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16;
                    U = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128;
                    V = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128;

                    // Writing the Y component
                    yuv420sp[yIndex++] = (byte) ((Y < 0) ? 0 : ((Y > 255) ? 255 : Y));

                    // Writing the U and V components
                    if (j % 2 == 0 && i % 2 == 0) {
                        yuv420sp[uvIndex++] = (byte) ((V < 0) ? 0 : ((V > 255) ? 255 : V));
                        yuv420sp[uvIndex++] = (byte) ((U < 0) ? 0 : ((U > 255) ? 255 : U));
                    }

                    index++;
                }
            }
        }


//        @Override
//        protected Void doInBackground(Void... voids) {
//            try {
//                //List<Bitmap> images = loadImages();  // 加载图像
//                //List<Bitmap> images = loadImagesInBackground();  // 在后台加载图像
//                List<Bitmap> images = loadImagesInBackground();  // 使用 MediaCodec 提取帧
//                Uri outputUri = createOutputUri();
//                ParcelFileDescriptor pfd = getContentResolver().openFileDescriptor(outputUri, "w");
//                FileOutputStream outputStream = new FileOutputStream(pfd.getFileDescriptor());
//                //encodeImagesToH264(images, outputStream);
//                encodeImagesToH264AndSendToServer(images);
//            } catch (IOException e) {
//                e.printStackTrace();
//                success = false;
//            }
//            return null;
//        }


//        private List<Bitmap> loadImages() {
//            List<Bitmap> images = new ArrayList<>();
//            try {
//                for (int i = 1; i <= 30; i++) {
//                    String fileName = String.format(Locale.US, "%03d.png", i);
//                    InputStream is = getAssets().open(fileName);
//                    Bitmap bitmap = BitmapFactory.decodeStream(is);
//                    images.add(bitmap);
//                    is.close();
//                }
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//            return images;
//        }



//        //新方法：异步加载图像
//        private List<Bitmap> loadImagesInBackground() throws IOException {
//            final List<Bitmap> images = new ArrayList<>();
//            final MediaMetadataRetriever retriever = new MediaMetadataRetriever();
//
//            // 创建一个新的线程加载图像
//            Thread loadingThread = new Thread(new Runnable() {
//                @Override
//                public void run() {
//                    try {
//                        // 从 assets 文件夹加载视频文件
//                        AssetFileDescriptor afd = getAssets().openFd("Jockey_000_270.mp4"); // 使用你的MP4文件名
//                        retriever.setDataSource(afd.getFileDescriptor(), afd.getStartOffset(), afd.getLength()); // 设置数据源
//
//                        // 获取视频的时长（单位是毫秒，转换为微秒）
//                        String durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
//                        long durationMs = Long.parseLong(durationStr);  // 时长是毫秒
//                        long duration = durationMs * 1000;  // 转换为微秒
//
//                        // 设置帧间隔，假设我们每秒提取30帧
//                        long frameInterval = 1000000 / 30;  // 1秒 = 1000000微秒，30帧/秒
//
//                        // 提取每一帧
//                        for (long time = 0; time <= duration; time += frameInterval) {
//                            Bitmap bitmap = retriever.getFrameAtTime(time, MediaMetadataRetriever.OPTION_CLOSEST);  // 获取指定时间的帧
//                            if (bitmap != null) {
//                                images.add(bitmap);
//                            }
//                        }
//                        retriever.release();
//                    } catch (IOException e) {
//                        e.printStackTrace();
//                    }
//                }
//            });
//
//            loadingThread.start();  // 启动线程
//
//            try {
//                loadingThread.join();  // 等待线程执行完毕
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//
//            return images;
//        }



//        private List<Bitmap> loadImagesWithFFmpeg_2() throws IOException {
//            final List<Bitmap> images = new ArrayList<>();
//            final String videoFileName = "Jockey_000_270.mp4";
//            final File outputDir = new File(getExternalFilesDir(null), "extracted_frames");  // 输出目录
//
//            // 确保输出目录存在
//            if (!outputDir.exists()) {
//                outputDir.mkdirs();
//            }
//            // 拷贝视频文件到缓存目录
//            String videoFilePath = copyVideoFromAssets(videoFileName);
//            // FFmpeg 命令：提取视频帧为图片文件
//            String command = "-i " + videoFilePath + " -vf fps=30 " + outputDir.getAbsolutePath() + "/frame_%04d.png";
//
//            // 执行 FFmpeg 命令
//            FFmpeg.executeAsync(command, new ExecuteCallback() {
//                @Override
//                public void apply(long executionId, int returnCode) {
//                    if (returnCode == Config.RETURN_CODE_SUCCESS) {
//                        Log.d("FFmpeg意", "视频帧提取成功");
//
//                        // 提取的帧保存完成后，将其加载为 Bitmap
//                        File[] frameFiles = outputDir.listFiles();
//                        if (frameFiles != null && frameFiles.length > 0) {
//                            Log.d("FFmpeg意", "提取的帧文件数量：" + frameFiles.length);
//                            Arrays.sort(frameFiles);  // 确保按顺序加载
//                            for (File frameFile : frameFiles) {
//                                Bitmap bitmap = BitmapFactory.decodeFile(frameFile.getAbsolutePath());
//                                if (bitmap != null) {
//                                    images.add(bitmap);
//                                    Log.d("FFmpeg意", "加载帧: " + frameFile.getName());
//                                } else {
//                                    Log.e("FFmpeg意", "无法加载帧: " + frameFile.getName());
//                                }
//                            }
//                        } else {
//                            Log.e("FFmpeg意", "没有提取到任何帧文件！");
//                        }
//                    } else {
//                        Log.e("FFmpeg意", "视频帧提取失败，错误码：" + returnCode);
//                    }
//                }
//            });
//
//            // 等待 FFmpeg 执行完毕（如果需要同步执行）
//            while (images.isEmpty()) {
//                try {
//                    Thread.sleep(100);  // 简单等待，防止线程阻塞
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                }
//            }
//
//            // 如果没有成功提取任何图像，则返回一个空列表
//            if (images.isEmpty()) {
//                Log.e("FFmpeg", "视频帧提取失败，未提取到任何帧！");
//            }
//
//            return images;
//        }




//        private void encodeImagesToH264AndSendToServer(List<Bitmap> images) throws IOException {
//            new Thread(new Runnable() {
//                @Override
//                public void run() {
//                    // 使用MediaCodec创建编码器
//                    MediaCodec codec = null;
//                    try {
//                        codec = MediaCodec.createEncoderByType("video/avc");
//                        MediaFormat format = MediaFormat.createVideoFormat("video/avc", 960, 540);  //480, 270
//                        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
//                        format.setInteger(MediaFormat.KEY_BIT_RATE, BIT_RATE);
//                        format.setInteger(MediaFormat.KEY_FRAME_RATE, FRAME_RATE);
//                        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL);
//                        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
//                        codec.start();
//
//                        // 获取输入和输出缓冲区
//                        ByteBuffer[] inputBuffers = codec.getInputBuffers();
//                        ByteBuffer[] outputBuffers = codec.getOutputBuffers();
//                        MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
//
//                        long pts = 0;
//                        long frameTime = 1000000 / FRAME_RATE;  // 计算帧时间
//
//                        // 遍历图片列表并进行编码
//                        for (Bitmap bitmap : images) {
//                            byte[] yuvData = getYUV420FromBitmap(bitmap);
//                            if (yuvData == null || yuvData.length == 0) {
//                                Log.e("FFmpeg-1", "YUV data is empty or null for bitmap: " + bitmap);
//                                continue;
//                            }
//
//                            // 获取输入缓冲区并填充数据
//                            int inputBufferIndex = codec.dequeueInputBuffer(TIMEOUT_US);
//                            if (inputBufferIndex >= 0) {
//                                ByteBuffer inputBuffer = inputBuffers[inputBufferIndex];
//                                inputBuffer.clear();
//                                inputBuffer.put(yuvData);
//                                codec.queueInputBuffer(inputBufferIndex, 0, yuvData.length, pts, 0);
//                                pts += frameTime;
//                            }
//
//                            // 获取输出缓冲区并处理编码后的帧数据
//                            int outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US);
//                            while (outputBufferIndex >= 0) {
//                                ByteBuffer outputBuffer = outputBuffers[outputBufferIndex];
//                                byte[] frameData = new byte[bufferInfo.size];
//
//                                outputBuffer.get(frameData);
//                                // 打印每一帧的大小
//                                Log.d("FFmpeg", "编码帧数据大小: " + bufferInfo.size + " 字节");
//
//                                // 发送编码后的帧到服务器
//                                //mClient.sendFrame(frameData);  // 使用你提供的 sendFrame 方法发送帧数据
//
//
//
//                                codec.releaseOutputBuffer(outputBufferIndex, false);
//                                outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US);
//                            }
//                        }
//
//                        // 停止并释放编码器
//                        codec.stop();
//                        codec.release();
//
//                    } catch (IOException e) {
//                        Log.e("EncodingError", "Error while encoding video", e);
//                    } finally {
//                        // 确保编码器被释放
//                        if (codec != null) {
//                            codec.stop();
//                            codec.release();
//                        }
//                    }
//                }
//            }).start(); // 启动新线程
//        }
//
//
//
//
//        private void encodeImagesToH264(List<Bitmap> images, FileOutputStream outputStream) throws IOException {
//            MediaCodec codec = MediaCodec.createEncoderByType("video/avc");
//            MediaFormat format = MediaFormat.createVideoFormat("video/avc", 480, 270);
////            MediaCodec codec = MediaCodec.createEncoderByType("video/hevc");
////            MediaFormat format = MediaFormat.createVideoFormat("video/hevc", 480, 270);
//            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
//            format.setInteger(MediaFormat.KEY_BIT_RATE, BIT_RATE);
//            format.setInteger(MediaFormat.KEY_FRAME_RATE, FRAME_RATE);
//            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL);
//            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
//            codec.start();
//            Log.e("yan","11");
//
//            ByteBuffer[] inputBuffers = codec.getInputBuffers();
//            ByteBuffer[] outputBuffers = codec.getOutputBuffers();
//            MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
//            Log.e("yan","12");
//
//            long pts = 0;
//            long frameTime = 1000000 / FRAME_RATE;
//            long encodingStart = System.currentTimeMillis();  // 记录编码开始时间
//            for (Bitmap bitmap : images) {
//                byte[] yuvData = getYUV420FromBitmap(bitmap);
//                if (yuvData == null || yuvData.length == 0) {
//                    Log.e("FFmpeg-1", "YUV data is empty or null for bitmap: " + bitmap);
//                }
//                int inputBufferIndex = codec.dequeueInputBuffer(TIMEOUT_US);
//                if (inputBufferIndex >= 0) {
//                    ByteBuffer inputBuffer = inputBuffers[inputBufferIndex];
//                    inputBuffer.clear();
//                    inputBuffer.put(yuvData);
//                    codec.queueInputBuffer(inputBufferIndex, 0, yuvData.length, pts, 0);
//                    pts += frameTime;
//                }
//                int outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US);
//                while (outputBufferIndex >= 0) {
//                    ByteBuffer outputBuffer = outputBuffers[outputBufferIndex];
//                    byte[] frameData = new byte[bufferInfo.size];
//
//                    outputBuffer.get(frameData);
//                    // 打印每一帧的大小
//                    Log.d("FFmpeg", "写入帧数据，帧大小: " + bufferInfo.size + " 字节");
//                    outputStream.write(frameData);
//                    codec.releaseOutputBuffer(outputBufferIndex, false);
//                    outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US);
//                }
//            }
//            long encodingEnd = System.currentTimeMillis();  // 记录编码结束时间
//
//
//            //codec.signalEndOfInputStream();
//
//            handleEndOfStream(codec, outputStream);
//            Log.e("yan","15");
//            codec.stop();
//            codec.release();
//            Log.e("yan","16");
//            long encodingTime = encodingEnd - encodingStart;  // 计算编码耗时
//            Log.d("Encoding666", "30images："+OUTPUT_FILE_NAME + " Encoding time: " + encodingTime + "ms");  // 记录或输出编码耗时
//
//        }



//        private void handleEndOfStream(MediaCodec codec, FileOutputStream outputStream) throws IOException {
//            MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
//            int outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US);
//            while (outputBufferIndex != MediaCodec.INFO_TRY_AGAIN_LATER) {
//                if (outputBufferIndex >= 0) {
//                    ByteBuffer outputBuffer = codec.getOutputBuffer(outputBufferIndex);  // 使用 getOutputBuffer 方法获取单个缓冲区
//
//                    // 检查是否到达流的末尾
//                    if ((bufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
//                        break;
//                    }
//
//                    if (bufferInfo.size > 0) {
//                        byte[] data = new byte[bufferInfo.size];
//                        outputBuffer.get(data);
//                        outputBuffer.clear();
//
//                        // 将编码后的数据写入文件
//                        outputStream.write(data, 0, bufferInfo.size);
//                    }
//                    codec.releaseOutputBuffer(outputBufferIndex, false);
//                } else if (outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
//                    // 如果输出格式发生改变，忽略。在大多数情况下，这意味着输出格式信息可用。
//                    // 输出格式变更逻辑可以在这里处理。
//                }
//
//                outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US);
//            }
//
//            // 清理工作，确保所有数据已写入文件
//            outputStream.flush();
//            outputStream.close();
//        }

    }




}