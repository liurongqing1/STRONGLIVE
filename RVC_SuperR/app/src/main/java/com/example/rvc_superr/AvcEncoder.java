package com.example.rvc_superr;

import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.util.Log;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

import static android.media.MediaCodec.BUFFER_FLAG_CODEC_CONFIG;
import static android.media.MediaCodec.BUFFER_FLAG_KEY_FRAME;


public class AvcEncoder
{
    private final static String TAG = "MeidaCodec";

    private int TIMEOUT_USEC = 12000;

    private MediaCodec mediaCodec;
    int m_width;
    int m_height;
    int m_framerate;
    Client mClient;

    public byte[] configbyte;


    public AvcEncoder(int width, int height, int framerate, int bitrate, Client mClient) {

        m_width  = width;
        m_height = height;
        m_framerate = framerate;
        this.mClient=mClient;
        MediaFormat mediaFormat = MediaFormat.createVideoFormat("video/avc", width, height);
        //MediaFormat mediaFormat = MediaFormat.createVideoFormat("video/hevc", width, height);
        mediaFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar);
        mediaFormat.setInteger(MediaFormat.KEY_BIT_RATE, width*height*5);
        mediaFormat.setInteger(MediaFormat.KEY_FRAME_RATE, 30);
        mediaFormat.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);
        try {
            mediaCodec = MediaCodec.createEncoderByType("video/avc");
            //mediaCodec = MediaCodec.createEncoderByType("video/hevc");
        } catch (IOException e) {
            e.printStackTrace();
        }
        //配置编码器参数
        mediaCodec.configure(mediaFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        //启动编码器
        mediaCodec.start();
    }

    private void StopEncoder() {
        if (mediaCodec != null) {
            mediaCodec.stop();
            mediaCodec.release();
            mediaCodec = null;
        }
    }

    public boolean isRuning = false;

    public void StopThread(){
        isRuning = false;
        StopEncoder();
    }

    int count = 0;

    public void StartEncoderThread(){
        Thread EncoderThread = new Thread(new Runnable() {

            @Override
            public void run() {
                isRuning = true;// 标识编码线程正在运行
                byte[] input = null;// 存储从队列中取出的数据
                long pts =  0;// 时间戳
                long generateIndex = 0;// 用来生成时间戳的索引
                long inputTimeNs=0; // 开始编码之前的时间戳
                long outputTimeNs=0; // 完成编码后的时间戳
                long inputTimeNs_2=0; // 开始YUV420格式转换之前的时间戳
                long outputTimeNs_2=0; // 完成YUV420格式转换后的时间戳
                long encodeTimeNs_1=0;
                long encodeTimeNs_2=0;
                //long outputTimeNs_3=0; //##############
                //long outputTimeNs_4=0; //##############
                while (isRuning) {
                    //访问MainActivity用来缓冲待解码数据的队列
                    // 检查是否有待编码的数据
                    if (MainActivity.YUVQueue.size() >0){
                        //从缓冲队列中取出一帧
                        input = MainActivity.YUVQueue.poll();
                        /****/
                        byte[] yuv420sp = new byte[m_width*m_height*3/2];
                        inputTimeNs_2 = System.nanoTime(); // 获取启动编码前的时间戳
                        //把待编码的视频帧转换为YUV420格式
                        NV21ToNV12(input,yuv420sp,m_width,m_height);
                        input = yuv420sp;
                        outputTimeNs_2 = System.nanoTime(); // 获取编码完成后的时间戳
                        // 计算编码耗时，单位为纳秒
                        encodeTimeNs_1 = outputTimeNs_2 - inputTimeNs_2;
                        // 将纳秒转换为毫秒，并进行输出日志
                        //outputTimeNs_3 = outputTimeNs_3+ encodeTimeNs_2;
                        Log.d("一帧编码的时间", "YUV420 Encoding time for frame " + generateIndex + ": " + encodeTimeNs_1 / 1000000L + " ms");
                    }
                    if (input != null) {
                        try {
                            // 获取输入缓冲区，用于传递数据到编码器
                            ByteBuffer[] inputBuffers = mediaCodec.getInputBuffers();
                            // 获取输出缓冲区，用于取出编码后的数据
                            ByteBuffer[] outputBuffers = mediaCodec.getOutputBuffers();
                            // 尝试获取可用的输入缓冲区索引
                            int inputBufferIndex = mediaCodec.dequeueInputBuffer(-1);
                            if (inputBufferIndex >= 0) {
                                pts = computePresentationTime(generateIndex);// 计算时间戳
                                ByteBuffer inputBuffer = inputBuffers[inputBufferIndex];
                                inputBuffer.clear();
                                //把转换后的YUV420格式的视频帧放到编码器输入缓冲区中
                                inputBuffer.put(input);// 将输入数据填充到输入缓冲区
                                inputTimeNs = System.nanoTime(); // 获取启动编码前的时间戳
                                // 将带有数据的输入缓冲区送回编码器，进行MediaCodec编码处理
                                mediaCodec.queueInputBuffer(inputBufferIndex, 0, input.length, pts, 0);
                                generateIndex += 1;
                            }

                            MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
                            int outputBufferIndex = mediaCodec.dequeueOutputBuffer(bufferInfo, TIMEOUT_USEC);

                            while (outputBufferIndex >= 0) {

                                ByteBuffer outputBuffer = outputBuffers[outputBufferIndex];
                                byte[] outData = new byte[bufferInfo.size];
                                outputTimeNs = System.nanoTime(); // 获取编码完成后的时间戳
                                // 计算编码耗时，单位为纳秒
                                encodeTimeNs_2 = outputTimeNs - inputTimeNs;
                                // 将纳秒转换为毫秒，并进行输出日志
                                long ii=generateIndex-1;
                                long encodeTimeNs_3 = encodeTimeNs_2 + encodeTimeNs_1;
                                long iii=bufferInfo.size;
                                Log.d("一帧编码的时间", "H264 Encoding time for frame " + ii + ": " + encodeTimeNs_2 / 1000000L + " ms"+"---size:"+bufferInfo.size);
                                Log.e("一帧编码的时间", "H264&&YUV420 Encoding time for frame " + ii + ": " + encodeTimeNs_3 / 1000000L + " ms");

                                outputBuffer.get(outData);// 将编码后的数据复制到outData数组中

                                if(bufferInfo.flags == BUFFER_FLAG_CODEC_CONFIG){
                                    configbyte = new byte[bufferInfo.size];
                                    configbyte = outData;// 编码器生成的SPS和PPS数据被保存下来
                                }else if(bufferInfo.flags == BUFFER_FLAG_KEY_FRAME){
                                    // 合并SPS/PPS和关键帧数据，然后一起发送
                                    byte[] keyframe = new byte[bufferInfo.size + configbyte.length];
                                    System.arraycopy(configbyte, 0, keyframe, 0, configbyte.length);// 将sps和pps填充到关键帧前面
                                    //把编码后的视频帧从编码器输出缓冲区中拷贝出来
                                    System.arraycopy(outData, 0, keyframe, configbyte.length, outData.length);
                                    // 通过客户端发送关键帧
                                    Log.e("666", Arrays.toString(keyframe));
                                    mClient.sendFrame(keyframe);// 发送带有SPS和PPS的关键帧
                                }else{
                                    // 通过客户端发送普通帧
                                    mClient.sendFrame(outData);// 发送非关键帧
                                }

                                mediaCodec.releaseOutputBuffer(outputBufferIndex, false);
                                outputBufferIndex = mediaCodec.dequeueOutputBuffer(bufferInfo, TIMEOUT_USEC);
                            }

                        } catch (Throwable t) {
                            t.printStackTrace();
                        }
                    } else {
                        try {
                            Thread.sleep(500);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        });
        EncoderThread.start();

    }

    private void NV21ToNV12(byte[] nv21,byte[] nv12,int width,int height){
        if(nv21 == null || nv12 == null)return;
        int framesize = width*height;
        int i = 0,j = 0;
        System.arraycopy(nv21, 0, nv12, 0, framesize);
        for(i = 0; i < framesize; i++){
            nv12[i] = nv21[i];
        }
        for (j = 0; j < framesize/2; j+=2)
        {
            nv12[framesize + j-1] = nv21[j+framesize];
        }
        for (j = 0; j < framesize/2; j+=2)
        {
            nv12[framesize + j] = nv21[j+framesize-1];
        }
    }

    /**
     * Generates the presentation time for frame N, in microseconds.
     */
    private long computePresentationTime(long frameIndex) {
        return 132 + frameIndex * 1000000 / m_framerate;
    }



}