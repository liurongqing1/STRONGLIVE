# 实时视频超分辨率重建项目 (SR_Android)

## 项目概述
本项目实现了针对移动端的实时视频超分辨率重建和MP4视频文件超分辨率重建，包含移动端、视频超分辨率重建服务器和桌面端三大模块。

## 模块

### 1. 移动端客户端（Android App）
项目位置：`RVC_SuperR`  
Android应用可以用于发送低分辨率的视频进行超分辨率处理。应用包括以下功能：

#### 功能
- **实时视频直播：**
  - 点击`开始直播`，打开摄像头采集视频同时连接超分辨率重建服务器。
  - 逐帧将视频流以H264编码发送给服务器。
  - 点击`结束直播`，断开与服务器的连接，关闭摄像头。

- **MP4视频文件超分辨率：**
  - 点击`connect`，与服务器建立连接。
  - 点击`start conversion`，将低分辨率MP4视频文件逐帧提取出来并进行h264编码发送给服务器。
  - 点击`disconnect`，断开与服务器的连接。

#### 关键文件
- **Client.java：**
  - 处理与超分辨率重建服务器的通信，包括连接、数据发送等。可以修改服务器的IP地址和端口号。

- **MainActivity.java：**
  - 实现实时视频超分辨率重建的逻辑。可以修改摄像头采集的分辨率（`width` 和 `height`）。

- **img_Activity.java：**
  - 实现MP4视频文件（该文件在`assets`资源文件夹下）的超分辨率重建。可以修改视频文件名，并根据视频分辨率修改`MediaFormat.createVideoFormat("video/avc", 960, 540)`中的`width`和`height`。

### 2. 桌面端客户端（接收视频服务器）
项目位置：`Video_sever`
桌面客户端接收来自超分辨率服务器的高分辨率视频帧，进行解码，并进行实时播放或保存为MP4文件。

#### 功能
- **实时播放：**
  - 对现有代码注释，将注释代码解注释。
  
- **保存为MP4文件：**
  - 使用`main.py`文件将解码后的视频帧保存为MP4文件。

#### 关键文件
- **main.py：**
  - 处理视频帧的接收和解码。

- **参数：**
  - `SERVER_HOST`：主机的IP地址（可以使用`ipconfig`命令获取）。
  - `SERVER_PORT`：自定义的端口号，用于与服务器进行通信。
  - `cv2.VideoWriter('output.mp4', fourcc, fps, (3840, 2160))`：修改宽度和高度与接收到的视频帧一致。

### 3. 视频超分辨率重建服务器
项目位置：`.../STRONGLIVE/Video_server/EDSR-PyTorch/src/main_7.py`和`.../STRONGLIVE/Video_server/EDSR-PyTorch/src/ trainer_end_6.py`  
该服务器处理来自移动端的视频帧，进行超分辨率重建，并将高分辨率的帧返回给桌面客户端。

#### 功能
- **接收移动端发送的H264帧：**
  - 解码接收到的帧并保存为`input.mp4`文件。
  
- **超分辨率处理：**
  - 对解码后的帧进行预处理，并存入队列中进行超分辨率重建。
  - 重建后进行H264编码，并存储为`output.h264`文件。

- **发送帧给桌面端：**
  - 将超分辨率处理后的帧发送给桌面客户端进行播放或保存。

#### 关键文件
- **main_7.py：**
  - 负责解码接收到的H264帧并保存为`input.mp4`文件。
  - 对解码后的帧进行预处理
  - 配置参数：
    - 修改服务器的IP和端口号，使其与Android端一致。
    - 修改`cv2.VideoWriter()`视频帧的分辨率（与接收的视频帧一致，即与Android端发送的视频帧一致）。

- **trainer_end_6.py：**
  - 处理与桌面端的通信。
  - 进行超分辨率处理和帧的H264编码。
  - 配置参数：
    - 修改连接桌面端服务器的IP和端口号，使其与桌面端一致。
    - 修改`nvc.PyNvEncoder()`、`SamplePyTorch.cconverter()`、`nvc.PyNvDecoder()`的分辨率参数（输出帧的分辨率为移动端视频的4倍）。

## 执行步骤
#### 步骤1：启动桌面端客户端
启动桌面端客户端，接收来自超分辨率服务器的高分辨率视频帧。
#### 步骤2：启动超分辨率服务器
1. **激活环境：**
   ```bash
   source /home/srteam/anaconda3/bin/activate
   source activate
   conda activate /home/srteam/anaconda3/envs/VMAF2
   source /home/srteam/lrq/switch-cuda.sh 11.3
2. **运行超分辨率服务器：**
   ```bash
   python .../STRONGLIVE/Video_server/EDSR-PyTorch/src/main_7.py --model FMEN2 --skip 1 --scale 4 --save Jockey_007 --save_results --dir_demo Jockey_007 --data_test Jockey_007 --data_range 1-30 --pre_train .../STRONGLIVE/Video_server/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
#### 步骤3：运行Android应用
打开Android应用，选择实时视频直播或MP4文件进行转换。

   
 
   
