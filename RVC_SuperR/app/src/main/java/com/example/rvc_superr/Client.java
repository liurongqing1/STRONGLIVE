package com.example.rvc_superr;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketAddress;
import java.nio.ByteBuffer;

import android.util.Log;

public class Client {
    private static Client mClient;
    private static Socket mSocket;
    // 要连接的服务器IP地址和端口号
    private static final String HOST_ADDRESS ="10.8.50.249";//”192.168.133.85“"10.138.235.15";//"10.138.238.127";//"10.8.50.249" ;//"10.8.50.249";//"10.138.137.131";//"83g04486n4.vicp.fun";//"192.168.10.1";//"151.101.194.49";////;
    private static final int HOST_PORT = 12582
            ;//12580;//443;//12580;

    // 获取Client的单例
    public static Client getInstance() {
        if (mClient == null) {
            mClient = new Client();
        }
        return mClient;
    }

    // 在子线程中打开Socket通道,连接服务器
    public void connect() {
        new Thread(() -> {
            if (mSocket == null) {
                try {
                    SocketAddress socketAddress = new InetSocketAddress(HOST_ADDRESS, HOST_PORT);
                    mSocket = new Socket();
                    mSocket.connect(socketAddress, 5000);
                    Log.i("Client-2", "socket成功连接");
                } catch (IOException e) {
                    Log.e("Client-2", "连接失败", e);
                    e.printStackTrace();
                }
            }
        }).start();
    }

    // 断开连接
    public void disconnect() {
        if (mSocket != null && mSocket.isConnected()) {
            try {
                mSocket.close();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                mSocket = null;
                mClient = null;
            }
        }
    }

    // 在子线程中发送帧数据
    public void sendFrame(byte[] frameData) {
        // 计算出帧数据大小
        int frameSize = frameData.length;
        // 将帧大小转换为4字节数组 (big-endian)
        byte[] frameSizeBytes = ByteBuffer.allocate(4).putInt(frameSize).array();

        // 创建一个新数组，包含帧大小和帧数据
        byte[] combined = new byte[4 + frameSize];
        System.arraycopy(frameSizeBytes, 0, combined, 0, 4);
        System.arraycopy(frameData, 0, combined, 4, frameSize);

        // 使用现有方法发送合并后的数据
        sendData(combined);
    }

    // 一个通用的发送数据的方法
    private void sendData(byte[] bytes) {
        new Thread(() -> {
            if (mSocket != null && mSocket.isConnected()) {
                try {
                    OutputStream os = mSocket.getOutputStream();
                    os.write(bytes);
                    os.flush();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }
}