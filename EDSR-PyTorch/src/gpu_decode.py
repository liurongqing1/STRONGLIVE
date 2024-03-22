import cv2
import ffmpegcv
import numpy as np
import torch
import torchvision
import time
import tqdm

#print(cv2.cuda.getCudaEnabledDeviceCount())
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
def is_cuda(data):
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            print("数据在 GPU 上")
        else:
            print("数据在 CPU 上")

# 视频输入参数
input_video_path = "/home/srteam/lrq/EDSR-PyTorch/gpucode/Jockey_000/Jockey_000540.mp4"

def decode_ff(input_video_path):
    '''    # 创建CUDA加速的解码器
    # decoder = cv2.cuda.createH264Decoder()
    cap = ffmpegcv.VideoCaptureNV(input_video_path)  # NVIDIA GPU0
    Frame = []
    for frame in cap:
    #while True:
        #ret, frame = cap.read()
        if frame is None:
            break
    Frame.append(frame)
        #print(np.array(frame).shape) #（540，960 ，3）
    Frame = torch.Tensor(Frame)
    Frame = Frame.permute(0, 3, 1, 2)
    #print(np.array(Frame).shape) #(30, 3, 540, 960)
    cap.release()
    '''
    # 代理任何 VideoCapture&VideoWriter 的参数和kargs
    t0 = time.time()
    vid_noblock = ffmpegcv.noblock(ffmpegcv.VideoCaptureNV, input_video_path, pix_fmt='rgb24')
    print("decode time: " + str(time.time()-t0))#555ms
    # 这很快
    def gpu_tense():
        time.sleep(0.01)
    tg=0
    tc=0
    tr = 0
    for _ in tqdm.trange(30):
        t5 = time.time()
        ret, frame = vid_noblock.read()  # 当前图像已经被缓冲，不会占用时间
        tr=tr+time.time() - t5
        print("read time: " + str(tr)+"\n")  # 103ms
        #print(frame.type)
        #print(np.array(frame).shape)

        frame = torch.Tensor(frame)
        is_cuda(frame)#cpu
        Frame=frame.reshape(1,540,960,3)
        t3 = time.time()
        Frame = torch.Tensor(Frame).to('cuda')#!!!!!!!!!
        tg=tg + time.time() - t3
        print("cpu->gpu time: "+str(tg))#856ms

        t4 = time.time()
        Frame = torch.Tensor(Frame).to('cpu')#!!!!!!!!!
        tc=tc + time.time() - t4
        print("gpu->cpu time: "+str(tc))#38ms

        Frame = Frame.permute(0, 3, 1, 2)
        is_cuda(Frame)#gpu
        #print(np.array(Frame).shape)
        #gpu_tense()  # 同时，下一帧在后台缓冲
    vid_noblock.release()
    
    return 0

def decode_cv(input_video_path):
    # 打开视频文件
    #cap = cv2.cudacodec.createVideoReader(input_video_path)
    cap = cv2.cuda.VideoReader(input_video_path)
    Frame = []
    while cap.isOpened():
        # 读取一帧
        ret, frame = cap.nextFrame()
        if not ret:
            break
        Frame.append(frame)
    Frame = torch.Tensor(Frame)
    Frame = Frame.permute(0, 3, 1, 2)
    print(np.array(Frame).shape)
    # 使用CUDA加速的超分辨率处理帧
    # sr = self.model(lr, idx_scale)
    cap.release()
    cv2.destroyAllWindows()
    return 0

def decode_torch(input_video_path):
    # 打开视频文件
    reader = torchvision.io.VideoReader(input_video_path, "video", num_threads=0, device='cuda:0')
    Frame = []
    for frame in reader:
        Frame.append(frame["data"])
    Frame = torch.Tensor(Frame)
    Frame = Frame.permute(0, 3, 1, 2)
    print(np.array(Frame).shape)
    # 使用CUDA加速的超分辨率处理帧
    # sr = self.model(lr, idx_scale)
    cap.release()
    cv2.destroyAllWindows()
    return 0

def decode_ff(input_video_path):
    '''    # 创建CUDA加速的解码器
    # decoder = cv2.cuda.createH264Decoder()
    cap = ffmpegcv.VideoCaptureNV(input_video_path)  # NVIDIA GPU0
    Frame = []
    for frame in cap:
    #while True:
        #ret, frame = cap.read()
        if frame is None:
            break
    Frame.append(frame)
        #print(np.array(frame).shape) #（540，960 ，3）
    Frame = torch.Tensor(Frame)
    Frame = Frame.permute(0, 3, 1, 2)
    #print(np.array(Frame).shape) #(30, 3, 540, 960)
    cap.release()
    '''
    # 代理任何 VideoCapture&VideoWriter 的参数和kargs
    t0 = time.time()
    #vid_noblock = ffmpegcv.noblock(ffmpegcv.VideoCaptureNV, input_video_path, pix_fmt='rgb24')
    vid_noblock = ffmpegcv.noblock(ffmpegcv.VideoCapture, input_video_path, pix_fmt='rgb24')
    print("decode time: " + str(time.time()-t0))#555ms
    # 这很快
    def gpu_tense():
        time.sleep(0.01)

    tg=0
    tc=0
    tt = 0
    tp = 0
    tr = 0
    t7=0
    t8 = 0
    t9 = 0
    t10= 0

    t9 = time.time()
    for _ in tqdm.trange(30):
        t5 = time.time()
        ret, frame = vid_noblock.read()  # 当前图像已经被缓冲，不会占用时间
        tr=tr+time.time() - t5
        print("read time: " + str(tr))  # 103ms
        #print(np.array(frame).shape)

        t6 = time.time()
        with torch.no_grad():
            print(frame)
            frame = torch.from_numpy(frame).pin_memory()
            Frame = frame.to('cuda')
            print(frame.shape)
            Frame = Frame.permute(2, 0, 1)
            Frame = Frame.unsqueeze(0)
            #Frame = Frame.view(1, 540, 960, 3)
            #Frame = Frame.permute(0, 3, 1, 2)
            print(Frame)
            tg = tg + time.time() - t6
        print("trans time: " + str(tg) + "\n")  # 103ms

    print("all time: " + str(time.time() - t9) + "\n")  # 103ms

    '''
        t6 = time.time()
        frame = torch.Tensor(frame)
        ##is_cuda(frame)#cpu
        Frame=frame.reshape(1,540,960,3)
        tt = tt + time.time() - t5
        print("trans time: " + str(tt))  # 103ms

        t3 = time.time()
        Frame = torch.Tensor(Frame).to('cuda')#!!!!!!!!!
        tg=tg + time.time() - t3
        print("cpu->gpu time: "+str(tg))#856ms
    '''
    '''
        t4 = time.time()
        Frame = torch.Tensor(Frame).to('cpu')#!!!!!!!!!
        tc=tc + time.time() - t4
        print("gpu->cpu time: "+str(tc))#38ms

        t4 = time.time()
        Frame = Frame.permute(0, 3, 1, 2)
        tp = tp + time.time() - t4
        print("reshape time: " + str(tp))  # 103ms
        ##is_cuda(Frame)#gpu
        #print(np.array(Frame).shape)
        #gpu_tense()  # 同时，下一帧在后台缓冲
    '''
    vid_noblock.release()
    return 0


def encode_ff(input_video_path):
    # 代理任何 VideoCapture&VideoWriter 的参数和kargs
    t0 = time.time()
    #out_gpu = ffmpegcv.VideoWriterNV(input_video_path, 'h264', 30)
    vid_noblock = ffmpegcv.noblock(ffmpegcv.VideoWriter, input_video_path, 'h264', 30)
    print("encode time: " + str(time.time()-t0))#
    tg=0
    tc=0
    tr = 0
    for _ in tqdm.trange(30):

        frame = torch.Tensor(frame)
        is_cuda(frame)#cpu
        Frame=frame.reshape(540,960,3)
        Frame = Frame.permute(0, 3, 1, 2)

        Frame = torch.Tensor(Frame).to('cuda')#!!!!!!!!!


        is_cuda(Frame)#gpu
        out.write(frame1)
    out.release()
    return 0


t1 = time.time()
decode_ff(input_video_path)
t2 = time.time()
t = t2-t1
print("时间开销："+str(t)+'s')