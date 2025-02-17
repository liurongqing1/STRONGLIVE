import socket
import struct
import threading
import queue
import traceback
import numpy as np
import av  # 使用 PyAV 进 行 H.264 解码
# 从现有库中导入 args 配置，该配置在项目现有模块中已定义
from option import args
import utility
import model
import torch
import os
import cv2


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
from trainer_end_6 import Trainer  # 假设已从 trainer.py 导入

# 本服务器的IP和端口
SERVER_HOST = '10.8.50.249'
SERVER_PORT = 12582

# H.264解码帧缓冲区
decoded_frame_queue = queue.Queue(maxsize=200)
send_queue = queue.Queue(maxsize=50)

frame_counter = 0  # 全局帧计数器


def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def receive_frames(client_socket):
    """接收实时帧并存入接收缓冲区同时保存到 MP4 文件"""
    global frame_counter
    print(f"客户端 {client_socket.getpeername()} 已连接。")

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    fps = 30  # 假设帧率为 30
    output_file = cv2.VideoWriter('/home/srteam/lrq/EDSR-PyTorch/gpucode/Jockey_000/input.mp4', fourcc, fps, (960, 540))  # 请根据实际帧尺寸修改


    # 初始化解码器
    decoder = av.CodecContext.create('h264', 'r')

    try:

        while True:
            frame_size_bytes = recvall(client_socket, 4)
            if not frame_size_bytes:
                break

            frame_size = struct.unpack('!I', frame_size_bytes)[0]
            frame_data = recvall(client_socket, frame_size)
            if not frame_data:
                break

            frame_counter += 1

            try:
                # 解码 H.264 帧
                packet = av.Packet(frame_data)
                frames = decoder.decode(packet)

                for frame in frames:
                    # 将解码后的帧（numpy 格式）存入缓冲区
                    img = frame.to_ndarray(format='bgr24')
                    img_MP4 = frame.to_ndarray(format='rgb24')

                    print(f"未处理处理帧编号: {frame_counter}, 超分辨率张量尺寸: {img.shape}")

                    input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(
                        torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        dtype=torch.float32
                    )
                    if decoded_frame_queue.full():
                        print(f"解码帧缓冲区已满，丢弃帧编号: {frame_counter}")
                        continue

                    decoded_frame_queue.put((frame_counter, input_tensor), timeout=1)
                    print(f"帧编号: {frame_counter} 转换为张量并存入缓冲区")

                    # 将图像写入 MP4 文件
                    output_file.write(img_MP4)
                    print(f"已写入帧到 MP4 文件: 帧编号 {frame_counter}")

            except av.error.InvalidDataError as e:
                print(f"解码失败，帧编号: {frame_counter}, 错误: {e}")
                continue

    except Exception as e:
        print(f"接收或解码帧时发生错误: {e}")
        traceback.print_exc()
    finally:
        print(f"关闭从 {client_socket.getpeername()} 的连接。")
        client_socket.close()
        output_file.release()  # 释放 VideoWriter 对象以确保视频文件写入完毕


def main():
    if args.data_test == ['video']:
        from videotester import VideoTester
        model_inst = model.Model(args, checkpoint)
        t = VideoTester(args, model_inst, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            model_inst = model.Model(args, checkpoint)
            trainer = Trainer(args, None, model_inst, None, checkpoint,  decoded_frame_queue, send_queue)

            # 启动超分辨率处理线程
            processing_thread = threading.Thread(target=trainer.process_frames, daemon=True)
            processing_thread.start()


            # 启动发送帧线程
            sending_thread = threading.Thread(target=trainer.send_frames, daemon=True)
            sending_thread.start()

            # 启动服务器监听线程
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((SERVER_HOST, SERVER_PORT))
            server_socket.listen(1)
            print(f"监听地址 {SERVER_HOST}:{SERVER_PORT}，等待连接...")

            try:
                while True:
                    client_socket, client_address = server_socket.accept()
                    # 启动接收帧线程
                    client_thread = threading.Thread(
                        target=receive_frames,
                        args=(client_socket,),
                        daemon=True
                    )
                    client_thread.start()
            except Exception as e:
                print(f"服务器错误: {e}")
            finally:
                server_socket.close()


if __name__ == "__main__":
    main()