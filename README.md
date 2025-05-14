# STRONGLIVE:Adaptive Offloading and Scene-Aware SR Learning for 4K Live Streaming on Mobile Devices


## Project Overview
This project realizes real-time video super-resolution reconstruction and MP4 video file super-resolution reconstruction for mobile terminal, including three modules: mobile terminal, video super-resolution reconstruction server and desktop terminal.

## module 

### 1. Mobile Client（Android App）
Project Location：`RVC_SuperR`  

Android application can be used to send low resolution videos for super resolution processing. The application includes the following features:

#### functionality
- **Live video streaming：**
  - Click `Start Live Streaming` and turn on the camera to capture the video while connecting to the super-resolution reconstruction server.
  - - Send the video stream to the server in H264 encoding frame by frame.
  - Click `End Live Stream` to disconnect from the server and turn off the camera.

- **MP4 video file super resolution:**
  - Click `connect` to establish a connection with the server.
  - Click `start conversion` to extract the low resolution MP4 video file frame by frame and send it to the server with h264 encoding.
  - Click `disconnect` to disconnect from the server.
  
#### Key documents
- **Client.java：**
  - Handles communication with the Super Resolution Reconstruction Server, including connections, data sending, etc. Can modify the IP address and port number of the server.

- **MainActivity.java：**
  - Logic to implement super-resolution reconstruction of live video. The resolution (`width` and `height`) of the camera capture can be modified.

- **img_Activity.java：**
  - Enables super-resolution reconstruction of MP4 video files (which are in the `assets` resources folder). You can modify the video file name and modify the `width` and `height` in `MediaFormat.createVideoFormat(“video/avc”, 960, 540)` according to the video resolution.

### 2. Desktop client (receiving video server)
Project Location：`Video_sever`

The desktop client receives high-resolution video frames from the super-resolution server, decodes them, and either plays them in real time or saves them as MP4 files.


#### functionality
- **Real-time playback：**
  - Annotate existing code, and unannotate the annotated code.
  
- **Save as MP4 file:**
  - Use the `main.py` file to save the decoded video frames as MP4 files.

#### Key documents
- **main.py：**
  - Handles the reception and decoding of video frames.

- **parameters：**
  - `SERVER_HOST`: IP address of the host (can be obtained using the `ipconfig` command).
  - `SERVER_PORT`: customized port number to communicate with the server.
  - `cv2.VideoWriter('output.mp4', fourcc, fps, (3840, 2160))`: modify the width and height to match the received video frame.

### 3. Video Super Resolution Reconstruction Server
Project Location：`.../STRONGLIVE/Video_server/EDSR-PyTorch/src/main_7.py` and `.../STRONGLIVE/Video_server/EDSR-PyTorch/src/ trainer_end_6.py`  

This server processes the video frames from the mobile, performs super-resolution reconstruction, and returns the high-resolution frames to the desktop client.

#### functionality
- **Receive H264 frames sent from the mobile:**
  - Decode received frames and save as `input.mp4` file.
  
- **Super resolution processing:**
  - Preprocess the decoded frames and store them in a queue for super-resolution reconstruction.
  - H264 encoding is performed after reconstruction and stored as `output.h264` file.

- **Send frames to the desktop side:**
  - Sends the super-resolution processed frames to the desktop client for playback or saving.

#### Key documents
- **main_7.py：**
  - Responsible for decoding the received H264 frames and saving them as `input.mp4` file.
  - Preprocess the decoded frames
  - Configuration parameters:
    - Modify the IP and port number of the server to make it consistent with the Android side.
    - Modify the resolution of `cv2.VideoWriter()` video frames (consistent with the received video frames, i.e. consistent with the video frames sent from Android side).

- **trainer_end_6.py：**
  - Handles communication with the desktop side.
  - Performs super-resolution processing and H264 encoding of frames.
  - Configuration parameters:
    - Modify the IP and port number of the server connecting to the desktop side to match the desktop side.
    - Modify the resolution parameter of `nvc.PyNvEncoder()`, `SamplePyTorch.cconverter()`, `nvc.PyNvDecoder()` (the resolution of the output frames is 4 times that of the video on the mobile side).

## Execution steps
#### Step 1: Launch the desktop client
Launch the desktop client to receive high-resolution video frames from the Super Resolution Server.

#### Step 2: Start the Super Resolution Server
1. **activation environment：**
   ```bash
   source /home/srteam/anaconda3/bin/activate
   source activate
   conda activate /home/srteam/anaconda3/envs/VMAF2
   source /home/srteam/lrq/switch-cuda.sh 11.3
2. **Running a super-resolution server：**
   ```bash
   python .../STRONGLIVE/Video_server/EDSR-PyTorch/src/main_7.py --model FMEN2 --skip 1 --scale 4 --save Jockey_007 --save_results --dir_demo Jockey_007 --data_test Jockey_007 --data_range 1-30 --pre_train .../STRONGLIVE/Video_server/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only

#### Step 3：Run the Android app
Open the Android app and select live video streaming or MP4 file to convert

## Citation
 `
@misc{iwqos2025，
      title={STRONGLIVE:Adaptive Offloading and Scene-Aware SR Learning for 4K Live Streaming on Mobile Devices}, 
      author={Rongqing Liu and Yiqi Liu and Jie Ren and Ling Gao and Jie Zheng},
      year={2025},
      eprint={ },
      archivePrefix={ },
      primaryClass={ }
}
 `
   
 
   
