# import socket
# import struct
# import threading
# import av
# import traceback
# import cv2
# import numpy as np
# import time
#
# SERVER_HOST = '192.168.133.85'    ##'10.138.235.15'
# SERVER_PORT = 12580
#
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.bind((SERVER_HOST, SERVER_PORT))
# server_socket.listen(1)
# print(f"监听地址 {SERVER_HOST}:{SERVER_PORT}，等待连接...")
#
# def recvall(sock, n):
#     data = b''
#     try:
#         while len(data) < n:
#             packet = sock.recv(n - len(data))
#             if not packet:
#                 return None
#             data += packet
#     except Exception as e:
#         print(f"接收数据时出错: {e}")
#         return None
#     return data
#
# def show_frame(img):
#     # 设置目标宽度和高度
#     desired_width, desired_height = 1280, 720
#
#
#     # 确保图像是三通道的，并执行BGR到RGB的转换
#     if len(img.shape) == 2:  # 如果是单通道灰度图像，扩展成三通道
#         img = cv2.cvtColor(img, cv2.IMREAD_COLOR)
#     # 将BGR转换为RGB
#     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     rotated_img = cv2.rotate(rgb_img, cv2.ROTATE_180)  # OpenCV常量来实现180度旋转
#
#     # 创建窗口并设定窗口大小
#     cv2.namedWindow('Video Stream', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Video Stream', desired_width, desired_height)
#
#     # 显示旋转后的图像
#     cv2.imshow('Video Stream', rotated_img)
#
# def handle_client_connection(client_socket):
#     print(f"客户端 {client_socket.getpeername()} 已连接。")
#     codec = av.CodecContext.create('h264', 'r')
#
#     try:
#         while True:
#
#             frame_size_bytes = recvall(client_socket, 4)
#             if frame_size_bytes is None:
#                 print("客户端断开连接或未能发送数据。")
#                 break
#
#             frame_size = struct.unpack('!I', frame_size_bytes)[0]
#             print(f"接收帧大小: {frame_size}")
#
#             frame_data = recvall(client_socket, frame_size)
#             if frame_data is None:
#                 print("未能接收完整帧。")
#                 break
#
#
#             try:
#                 packet = av.Packet(frame_data)
#                 frames = codec.decode(packet)
#                 for frame in frames:
#                     img = frame.to_ndarray(format='bgr24')
#                     cv2.imshow('Video Stream', img)
#                     # show_frame(img)
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         return  # 用户手动关闭窗口
#             except av.error.InvalidDataError:
#                 print("解码阶段出现无效数据错误。跳过当前帧。")
#                 continue
#     except Exception as e:
#         traceback.print_exc()
#         print(f"发生错误: {e}")
#     finally:
#         print(f"关闭从 {client_socket.getpeername()} 的连接。")
#         codec.close()
#         cv2.destroyAllWindows()
#         client_socket.close()
#
# def main():
#     try:
#         while True:
#             client_socket, client_address = server_socket.accept()
#             client_thread = threading.Thread(
#                 target=handle_client_connection,
#                 args=(client_socket,),
#                 daemon=True
#             )
#             client_thread.start()
#     except Exception as e:
#         print(f"服务器错误: {e}")
#     finally:
#         server_socket.close()
#         cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()

#
import socket
import struct
import threading
import av
import traceback
import cv2
import numpy as np
import time

SERVER_HOST = '10.138.235.15'
SERVER_PORT = 12582

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(1)
print(f"监听地址 {SERVER_HOST}:{SERVER_PORT}，等待连接...")

def recvall(sock, n):
    data = b''
    try:
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
    except Exception as e:
        print(f"接收数据时出错: {e}")
        return None
    return data


def handle_client_connection(client_socket):
    print(f"客户端 {client_socket.getpeername()} 已连接。")
    codec = av.CodecContext.create('h264', 'r')

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    fps = 30  # 假设帧率为 30
    output_file = cv2.VideoWriter('output.mp4', fourcc, fps, (3840, 2160))  # 480, 270根据实际分辨率调整2560, 1440  1920, 1080

    try:
        while True:
            frame_size_bytes = recvall(client_socket, 4)
            if frame_size_bytes is None:
                print("客户端断开连接或未能发送数据。")
                break  # 客户端断开连接，停止接收数据

            frame_size = struct.unpack('!I', frame_size_bytes)[0]
            print(f"接收帧大小: {frame_size}")

            frame_data = recvall(client_socket, frame_size)
            if frame_data is None:
                print("未能接收完整帧。")
                break  # 客户端断开连接，停止接收数据

            try:
                packet = av.Packet(frame_data)
                frames = codec.decode(packet)
                for frame in frames:
                    # 将解码的帧转为 BGR 格式
                    img = frame.to_ndarray(format='bgr24')  # 先转为 BGR 格式
                    # 显示当前帧形状
                    print(f"当前帧 RGB 形状: {img.shape}")
                    # 保存帧到 MP4 文件
                    output_file.write(img)  # 写入当前帧
                    print(f"已写入帧到 MP4 文件")

            except av.error.InvalidDataError:
                print("解码阶段出现无效数据错误。跳过当前帧。")
                continue
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        output_file.release()  # 释放 VideoWriter 对象
        print(f"关闭从 {client_socket.getpeername()} 的连接。")
        codec.close()
        client_socket.close()  # 关闭客户端连接

def main():
    try:
        while True:
            client_socket, client_address = server_socket.accept()
            client_thread = threading.Thread(
                target=handle_client_connection,
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