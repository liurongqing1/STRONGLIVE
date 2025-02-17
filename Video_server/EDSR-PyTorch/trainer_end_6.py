import queue
import av
from option import args
import utility
import model
import os
import math
import imageio
from decimal import Decimal
import time
import struct  # 用于处理二进制数据
import socket  # 导入 socket 模块，这行可能遗漏了

import utility
import torch
import torch.nn as nn
import torch.nn.utils as utils
from tqdm import tqdm
import torch, gc  ###

import core

import imageio

import ffmpegcv
import cv2
import numpy as np

import tqdm
import PyNvCodec as nvc
import sys

sys.path.append(r"/home/srteam/lrq/VideoProcessingFramework/samples")
# print(sys.path)
import SamplePyTorch



class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp, decoded_frame_queue, send_queue):
        self.args = args
        self.scale = args.scale
        self.skip = args.skip
        self.ckp = ckp
        self.model = my_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.decoded_frame_queue = decoded_frame_queue
        self.send_queue = send_queue

        #2560x1440
        # 编码器初始化
        self.nvEnc = nvc.PyNvEncoder(
            {"codec": "h264", 's': '3840x2160', 'preset': 'P4', "profile": "high",
             "tuning_info": "high_quality", "bitrate": "10M", "fps": "30",
             "gop": str(self.skip)}, 0, nvc.PixelFormat.NV12
        )

        # 转换器初始化
        self.to_nv12 = SamplePyTorch.cconverter(3840, 2160, 0)  #2560, 1440  1920, 1080  1920x1080
        self.to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
        self.to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
        self.to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)

        # 服务器2的连接参数
        self.server_host = '10.138.235.15'
        self.server_port = 12582


        self.dstFile = open('/home/srteam/lrq/EDSR-PyTorch/gpucode/Jockey_000/output.h264', "wb")

        self.gpu_id = 0  # 确保选择正确的 GPU
        # 初始化解码器
        self.decoder = nvc.PyNvDecoder(3840, 2160, nvc.PixelFormat.NV12, nvc.CudaVideoCodec.H264, self.gpu_id)

    def connect_to_server(self):
        """尝试连接到服务器2，成功则返回 socket 对象"""
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.server_host, self.server_port))
            print(f"成功连接到服务器 {self.server_host}:{self.server_port}")
            return client_socket
        except Exception as e:
            print(f"无法连接到服务器 {self.server_host}:{self.server_port}: {e}")
            return None

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)

        return [_prepare(a) for a in args]



    def process_frames(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        """超分辨率处理并存入发送缓冲区"""
        while True:
            try:
                frame_id, input_tensor = self.decoded_frame_queue.get(timeout=1)

                # 模型超分辨率推理
                with torch.no_grad():
                    output = self.model(input_tensor, 0)
                print(f"处理帧编号: {frame_id}, 超分辨率张量尺寸: {output.shape}")


                # 转换为 NV12 格式
                sr_surface = self.to_nv12.run(SamplePyTorch.tensor_to_surface(output.squeeze(0).contiguous(), 0))

                #编码为 H.264 格式
                encFrame = np.ndarray(shape=(0), dtype=np.uint8)
                success_sr = self.nvEnc.EncodeSingleSurface(sr_surface, encFrame, sync=True)

                if not success_sr:
                    print(f"超分辨率重建失败，跳过帧编号: {frame_id}")
                    continue

                print(f"超分辨率处理完成，帧编号: {frame_id}, 大小"f": {len(encFrame)} bytes")

                # 将高分辨率帧放入队列
                self.send_queue.put((frame_id, encFrame), timeout=1)


                self.dstFile.write(bytearray(encFrame))
                print(f"已写入帧到 H264文件: 帧编号:{frame_id}, 大小: {len(encFrame)} bytes")





            except queue.Empty:
                continue
            except queue.Full:
                print(f"发送缓冲区已满，跳过帧编号: {frame_id}")
            except Exception as e:
                print(f"处理帧时发生错误: {e}")







    def send_frames(self):
        """从发送缓冲区中按序发送帧"""
        client_socket = self.connect_to_server()
        if client_socket is None:
            print("无法连接到服务器，退出发送线程。")
            return

        first_send = True
        while True:
            try:
                frame_id, frame_data = self.send_queue.get(timeout=1)
                # 首次发送延迟0.1秒
                if first_send:
                    time.sleep(0.1)
                    first_send = False

                frame_size_bytes = struct.pack('!I', len(frame_data))
                client_socket.sendall(frame_size_bytes)
                client_socket.sendall(frame_data)
                print(f"成功发送帧编号: {frame_id}, 大小: {len(frame_data)} bytes")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"发送时发生错误: {e}")
                client_socket.close()
                break

