import os
import math
import imageio
from decimal import Decimal
import time

import utility
import torch
import torch.nn as nn
import torch.nn.utils as utils
from tqdm import tqdm
import torch, gc###

import imageio
import ffmpegcv
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import concurrent.futures

import multiprocessing as mp
from multiprocessing import Pool, Process, current_process  #no!
import threading
#print(threading.active_count())#1
#print(threading.enumerate())
#print("Number of processors: ", mp.cpu_count())#16

from numba import cuda, jit
import PyNvCodec as nvc
import sys
sys.path.append(r"/home/srteam/lrq/VideoProcessingFramework/samples")
#print(sys.path)
import SamplePyTorch

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.skip = args.skip
        self.ckp = ckp
        self.loader_train = loader.loader_train#data.Data
        self.loader_test = loader.loader_test

        self.model = my_model
        self.loss = my_loss
        #待训练参数
        '''
                      'model.HFABs.0.bn1.weight', 'model.HFABs.0.bn1.bias', 'model.HFABs.0.bn2.weight',
                      'model.HFABs.0.bn2.bias', 'model.HFABs.0.bn3.weight', 'model.HFABs.0.bn3.bias',
                      'model.HFABs.0.squeeze.weight',
                      'model.HFABs.0.squeeze.bias', 'model.HFABs.0.convs.0.conv1.expand_conv.weight',
                      'model.HFABs.0.convs.0.conv1.expand_conv.bias',
                      'model.HFABs.0.convs.0.conv1.fea_conv.weight', 'model.HFABs.0.convs.0.conv1.fea_conv.bias',
                      'model.HFABs.0.convs.0.conv1.reduce_conv.weight',
                      'model.HFABs.0.convs.0.conv1.reduce_conv.bias', 'model.HFABs.0.convs.0.conv2.expand_conv.weight',
                      'model.HFABs.0.convs.0.conv2.expand_conv.bias',
                      'model.HFABs.0.convs.0.conv2.fea_conv.weight', 'model.HFABs.0.convs.0.conv2.fea_conv.bias',
                      'model.HFABs.0.convs.0.conv2.reduce_conv.weight', 'model.HFABs.0.convs.0.conv2.reduce_conv.bias',
                      'model.HFABs.0.excitate.weight', 'model.HFABs.0.excitate.bias',

                      'model.HFABs.1.bn1.weight', 'model.HFABs.1.bn1.bias', 'model.HFABs.1.bn2.weight',
                      'model.HFABs.1.bn2.bias',
                      'model.HFABs.1.bn3.weight', 'model.HFABs.1.bn3.bias', 'model.HFABs.1.squeeze.weight',
                      'model.HFABs.1.squeeze.bias',
                      'model.HFABs.1.convs.0.conv1.expand_conv.weight', 'model.HFABs.1.convs.0.conv1.expand_conv.bias',
                      'model.HFABs.1.convs.0.conv1.fea_conv.weight',
                      'model.HFABs.1.convs.0.conv1.fea_conv.bias', 'model.HFABs.1.convs.0.conv1.reduce_conv.weight',
                      'model.HFABs.1.convs.0.conv1.reduce_conv.bias',
                      'model.HFABs.1.convs.0.conv2.expand_conv.weight', 'model.HFABs.1.convs.0.conv2.expand_conv.bias',
                      'model.HFABs.1.convs.0.conv2.fea_conv.weight',
                      'model.HFABs.1.convs.0.conv2.fea_conv.bias', 'model.HFABs.1.convs.0.conv2.reduce_conv.weight',
                      'model.HFABs.1.convs.0.conv2.reduce_conv.bias', 'model.HFABs.1.excitate.weight',
                      'model.HFABs.1.excitate.bias',

                      'model.HFABs.2.bn1.weight', 'model.HFABs.2.bn1.bias', 'model.HFABs.2.bn2.weight',
                      'model.HFABs.2.bn2.bias',
                      'model.HFABs.2.bn3.weight', 'model.HFABs.2.bn3.bias', 'model.HFABs.2.squeeze.weight',
                      'model.HFABs.2.squeeze.bias', 'model.HFABs.2.convs.0.conv1.expand_conv.weight',
                      'model.HFABs.2.convs.0.conv1.expand_conv.bias',
                      'model.HFABs.2.convs.0.conv1.fea_conv.weight', 'model.HFABs.2.convs.0.conv1.fea_conv.bias',
                      'model.HFABs.2.convs.0.conv1.reduce_conv.weight', 'model.HFABs.2.convs.0.conv1.reduce_conv.bias',
                      'model.HFABs.2.convs.0.conv2.expand_conv.weight', 'model.HFABs.2.convs.0.conv2.expand_conv.bias',
                      'model.HFABs.2.convs.0.conv2.fea_conv.weight', 'model.HFABs.2.convs.0.conv2.fea_conv.bias',
                      'model.HFABs.2.convs.0.conv2.reduce_conv.weight', 'model.HFABs.2.convs.0.conv2.reduce_conv.bias',
                      'model.HFABs.2.excitate.weight', 'model.HFABs.2.excitate.bias',

                      'model.HFABs.3.bn1.weight', 'model.HFABs.3.bn1.bias', 'model.HFABs.3.bn2.weight',
                      'model.HFABs.3.bn2.bias',
                      'model.HFABs.3.bn3.weight', 'model.HFABs.3.bn3.bias', 'model.HFABs.3.squeeze.weight',
                      'model.HFABs.3.squeeze.bias', 'model.HFABs.3.convs.0.conv1.expand_conv.weight',
                      'model.HFABs.3.convs.0.conv1.expand_conv.bias',
                      'model.HFABs.3.convs.0.conv1.fea_conv.weight', 'model.HFABs.3.convs.0.conv1.fea_conv.bias',
                      'model.HFABs.3.convs.0.conv1.reduce_conv.weight',
                      'model.HFABs.3.convs.0.conv1.reduce_conv.bias', 'model.HFABs.3.convs.0.conv2.expand_conv.weight',
                      'model.HFABs.3.convs.0.conv2.expand_conv.bias',
                      'model.HFABs.3.convs.0.conv2.fea_conv.weight', 'model.HFABs.3.convs.0.conv2.fea_conv.bias',
                      'model.HFABs.3.convs.0.conv2.reduce_conv.weight',
                      'model.HFABs.3.convs.0.conv2.reduce_conv.bias', 'model.HFABs.3.excitate.weight',
                      'model.HFABs.3.excitate.bias',

                      'model.lr_conv.weight', 'model.lr_conv.bias',

        '''
        #'''冻结
        para_name = [ 'model.head.weight', 'model.head.bias',

                      'model.warmup.0.weight', 'model.warmup.0.bias',
                      'model.warmup.1.bn1.weight', 'model.warmup.1.bn1.bias', 'model.warmup.1.bn2.weight',
                      'model.warmup.1.bn2.bias', 'model.warmup.1.bn3.weight', 'model.warmup.1.bn3.bias',
                      'model.warmup.1.squeeze.weight', 'model.warmup.1.squeeze.bias','model.warmup.1.convs.0.conv1.expand_conv.weight',
                      'model.warmup.1.convs.0.conv1.expand_conv.bias', 'model.warmup.1.convs.0.conv1.fea_conv.weight',
                      'model.warmup.1.convs.0.conv1.fea_conv.bias', 'model.warmup.1.convs.0.conv1.reduce_conv.weight',
                      'model.warmup.1.convs.0.conv1.reduce_conv.bias', 'model.warmup.1.convs.0.conv2.expand_conv.weight',
                      'model.warmup.1.convs.0.conv2.expand_conv.bias', 'model.warmup.1.convs.0.conv2.fea_conv.weight',
                      'model.warmup.1.convs.0.conv2.fea_conv.bias', 'model.warmup.1.convs.0.conv2.reduce_conv.weight',
                      'model.warmup.1.convs.0.conv2.reduce_conv.bias', 'model.warmup.1.convs.1.conv1.expand_conv.weight',
                      'model.warmup.1.convs.1.conv1.expand_conv.bias', 'model.warmup.1.convs.1.conv1.fea_conv.weight',
                      'model.warmup.1.convs.1.conv1.fea_conv.bias', 'model.warmup.1.convs.1.conv1.reduce_conv.weight',
                      'model.warmup.1.convs.1.conv1.reduce_conv.bias', 'model.warmup.1.convs.1.conv2.expand_conv.weight',
                      'model.warmup.1.convs.1.conv2.expand_conv.bias', 'model.warmup.1.convs.1.conv2.fea_conv.weight',
                      'model.warmup.1.convs.1.conv2.fea_conv.bias', 'model.warmup.1.convs.1.conv2.reduce_conv.weight',
                      'model.warmup.1.convs.1.conv2.reduce_conv.bias', 'model.warmup.1.excitate.weight', 'model.warmup.1.excitate.bias',


                      'model.ERBs.0.conv1.expand_conv.weight','model.ERBs.0.conv1.expand_conv.bias','model.ERBs.0.conv1.fea_conv.weight',
                      'model.ERBs.0.conv1.fea_conv.bias','model.ERBs.0.conv1.reduce_conv.weight','model.ERBs.0.conv1.reduce_conv.bias',
                      'model.ERBs.0.conv2.expand_conv.weight','model.ERBs.0.conv2.expand_conv.bias','model.ERBs.0.conv2.fea_conv.weight',
                      'model.ERBs.0.conv2.fea_conv.bias','model.ERBs.0.conv2.reduce_conv.weight','model.ERBs.0.conv2.reduce_conv.bias',

                      'model.ERBs.1.conv1.expand_conv.weight','model.ERBs.1.conv1.expand_conv.bias','model.ERBs.1.conv1.fea_conv.weight',
                      'model.ERBs.1.conv1.fea_conv.bias','model.ERBs.1.conv1.reduce_conv.weight','model.ERBs.1.conv1.reduce_conv.bias',
                      'model.ERBs.1.conv2.expand_conv.weight', 'model.ERBs.1.conv2.expand_conv.bias','model.ERBs.1.conv2.fea_conv.weight',
                      'model.ERBs.1.conv2.fea_conv.bias','model.ERBs.1.conv2.reduce_conv.weight','model.ERBs.1.conv2.reduce_conv.bias',

                      'model.ERBs.2.conv1.expand_conv.weight', 'model.ERBs.2.conv1.expand_conv.bias',
                      'model.ERBs.2.conv1.fea_conv.weight','model.ERBs.2.conv1.fea_conv.bias',
                      'model.ERBs.2.conv1.reduce_conv.weight',
                      'model.ERBs.2.conv1.reduce_conv.bias', 'model.ERBs.2.conv2.expand_conv.weight',
                      'model.ERBs.2.conv2.expand_conv.bias',
                      'model.ERBs.2.conv2.fea_conv.weight', 'model.ERBs.2.conv2.fea_conv.bias',
                      'model.ERBs.2.conv2.reduce_conv.weight',
                      'model.ERBs.2.conv2.reduce_conv.bias',

                      'model.ERBs.3.conv1.expand_conv.weight', 'model.ERBs.3.conv1.expand_conv.bias',
                      'model.ERBs.3.conv1.fea_conv.weight', 'model.ERBs.3.conv1.fea_conv.bias',
                      'model.ERBs.3.conv1.reduce_conv.weight',
                      'model.ERBs.3.conv1.reduce_conv.bias', 'model.ERBs.3.conv2.expand_conv.weight',
                      'model.ERBs.3.conv2.expand_conv.bias',
                      'model.ERBs.3.conv2.fea_conv.weight', 'model.ERBs.3.conv2.fea_conv.bias',
                      'model.ERBs.3.conv2.reduce_conv.weight',
                      'model.ERBs.3.conv2.reduce_conv.bias',



                      'model.tail.0.weight', 'model.tail.0.bias'

                      ]#
        for name, parameter in self.model.named_parameters():
            # self.ckp.write_log("'{}',".format(name))
            if name in para_name:
                # self.ckp.write_log("{}".format(parameter))
                parameter.requires_grad = False
                #self.ckp.write_log("{}".format(str(i)))
            self.ckp.write_log('{}: {}'.format(name, parameter.requires_grad))
        #'''
        #优化器
        self.optimizer = utility.make_optimizer(args, self.model)#############

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))#ckp.dir='/home/srteam/lrq/EDSR-PyTorch', 'experiment', args.load

        self.error_last = 1e8

    def train(self):
        t0 = time.time()#######
        self.loss.step()######lr_scheduler.step
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)##
        for batch, (lr, hr, _,) in enumerate(self.loader_train):#[6, 3, 192, 192]
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)##############
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()#更新参数
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:##################
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            

            timer_data.tic()
        t1 = time.time()#######
        train_t=t1-t0
        self.ckp.write_log('Train time：{:.3f}s\n'.format(train_t))######

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()##########

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )

        self.model.eval()

        timer_test = utility.timer()
        diffm=0
        diffs = 0
        diffg = 0
        difft=0
        difff=0
        diffw=0

        diff1=0
        diff2=0
        diff3=0
        diff4=0


        Frame = []
        #vid_noblock = ffmpegcv.VideoWriterNV('/home/srteam/lrq/EDSR-PyTorch/gpucode/Jockey_000/a.mp4', 'h264', 30)############
        #vid_noblock = ffmpegcv.noblock(ffmpegcv.VideoWriterNV, '/home/srteam/lrq/EDSR-PyTorch/gpucode/Jockey_000/a.mp4', 'h264_nvenc', 30,  pix_fmt='rgb24')
        ####vid_noblock = ffmpegcv.noblock(ffmpegcv.VideoWriter, '', 'h264', 30, pix_fmt='rgb24')
        ###vw = cv2.VideoWriter('/home/srteam/lrq/EDSR-PyTorch/src/a.avi', cv2.VideoWriter_fourcc('H', '2', '6', '4'), 30, (2160,3840), True)

        dstFile = open('/home/srteam/lrq/EDSR-PyTorch/gpucode/Jockey_000/a.mp4', "wb")
        nvEnc = nvc.PyNvEncoder({ "codec": "h264",'s': '3840x2160', "profile": "high", 'preset': 'hq',
                                  "tuning_info": "high_quality", "bitrate": "10M", "fps":"30", "gop":"5"}, 0, nvc.PixelFormat.NV12)
        # Convert back to NV12
        to_nv12 = SamplePyTorch.cconverter(3840, 2160, 0)####
        to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
        to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
        to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)
        encFrame = np.ndarray(shape=(0), dtype=np.uint8)###
        framesReceived = 0
        framesFlushed = 0

        path = r'/home/srteam/lrq/EDSR-PyTorch/experiment/{}/test{}.txt'.format(self.args.data_test[0], self.scale[0])
        if self.args.save_results: self.ckp.begin_background()######
        #t0 = time.time() #resd start
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale): #0,4
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):#创建一个进度条来跟踪迭代的进度
                    t4 = time.time()  #######
                    lr, hr = self.prepare(lr, hr) #tensor().cuda
                    diff = time.time() - t4  ##########
                    diffg += diff
                    self.ckp.write_log('cpu->gpu: {:.3f}s\n'.format(diffg))
                    #print(lr,hr)
                    #'''
                    t0 = time.time()#######
                    #print(lr.shape) #torch.Size([1, 3, 540, 960])
                    sr = self.model(lr, idx_scale) #tensor().cuda (1,3,2160,3840)
                    #print(sr.shape)
                    diff = time.time() - t0##########
                    ##gc.collect()  #####
                    ##torch.cuda.empty_cache()  ###
                    #self.ckp.write_log('\nSR_one: {:.3f}s\n'.format(diff))
                    diffm += diff
                    self.ckp.write_log('SR: {:.3f}s\n'.format(diffm))
                    #'''
                    #'''

                    #跑通cpu线程,时间慢一点 trans:2944 wirte:209
                    '''####
                    width,height = sr.shape[2:]
                    yblock_size = 240
                    # 初始化输出帧
                    output_frame = np.zeros((width, height, 3), dtype=np.uint8)
                    # 创建并行执行的线程池
                    executor = ThreadPoolExecutor(max_workers=16)
                    # ProcessPoolExecutor(max_workers=9)
                    # 存储块编码任务的 Future 对象
                    block_futures = []
                    #all_task=[]
                    '''
                    gpu_id = 0  # GPU ID


                    t1 = time.time()  #######

                    #sr = sr.permute(0, 1, 2, 3).squeeze(0).contiguous()######
                    sr = sr.squeeze(0).contiguous()######torch.Size([3, 2160, 3840]
                    #print(sr.shape) #torch.Size([2160, 3840, 3])

                    surface_rgb = SamplePyTorch.tensor_to_surface(sr, gpu_id)
                    #print(surface_rgb)
                    # NV12格式的数据大小为（1.5 x 图像宽度 x 图像高度）字节
                    dst_surface = to_nv12.run(surface_rgb)
                    #print(dst_surface)
                    # Encoded video frame
                    success = nvEnc.EncodeSingleSurface(dst_surface, encFrame, sync=False) #False，则编码操作是异步的

                    #print(encFrame.device)
                    if success:
                        byteArray = bytearray(encFrame)
                        #print(encFrame,byteArray)
                        dstFile.write(byteArray)##前3帧出不来
                        framesReceived += 1

                    '''
                    #t2 = time.time()
                    for y in range(0, height, yblock_size):
                        # 提取当前块
                        block_futures.append((executor.submit(self.trans, sr[:, y:y + yblock_size, :]), y))
                    #diff = time.time() - t2  ##########
                    #diff2 += diff
                    #self.ckp.write_log('2: {:.3f}s\n'.format(diff2)) #51ms

                    #t3 = time.time()
                    torch.cuda.synchronize()  # Synchronize after the transfer
                    #diff = time.time() - t3  ##########
                    #diff3 += diff
                    #self.ckp.write_log('3: {:.3f}s\n'.format(diff3)) #1281ms

                    #t4 = time.time()
                    for future, y in block_futures:
                        if future.done():
                            output_frame[:, y:y + yblock_size, :] = future.result()
                        else:
                            block_futures.append((future, y))


                    #diff = time.time() - t4  ##########
                    #diff4 += diff
                    #self.ckp.write_log('4: {:.3f}s\n'.format(diff4)) #565ms

                    
                    #imageio.imwrite(f'/home/srteam/lrq/EDSR-PyTorch/gpucode/Jockey_000/{filename[0]}.png',output_frame)
                    #t = time.time()
                    #sr = sr.to('cpu').numpy()
                    #diff = time.time() - t  ##########
                    #diff1 += diff
                    #self.ckp.write_log('tocpu: {:.3f}s\n'.format(diff1))
                    
                    #orign
                    with torch.no_grad():
                        sr = sr.permute(0, 2, 3, 1).squeeze(0)
                        sr = sr.contiguous()
                        sr = sr.to('cpu').numpy()#, non_blocking=True).pin_memory()
                        #sr = np.array(sr)###时耗大
                        #Frame.append(sr)
                    #print(sr.shape)#
                    diff = time.time() - t1  ##########
                    difft += diff
                    self.ckp.write_log('trans: {:.3f}s\n'.format(difft))
                    
                                
                    #跑通cpu线程,时间慢一点 trans:2944 wirte:209
                    width,height = sr.shape[2:]
                    block_size = 240
                    # 初始化输出帧
                    output_frame = np.zeros((width, height, 3), dtype=np.uint8)
                    # 创建并行执行的线程池
                    executor = ThreadPoolExecutor() #需要启动线程池以及和线程池之间的通信，ThreadPoolExecutor 消耗的时间比单线程的版本还要慢！
                    # 存储块编码任务的 Future 对象
                    block_futures = []
                    all_task = []

                    t1 = time.time()  #######
                    
                    for y in range(0, height, block_size):
                        for x in range(0, width, block_size):
                            #print(y,x)
                            # 提取当前块
                            block = sr[:, :, x:x + block_size, y:y + block_size]
                            #print(block.shape)

                            #并行trans
                            #pool = Pool(processes=9)
                            #result = pool.apply_async(self.trans, all_task)#太慢

                            future = executor.submit(self.trans, block)
                            block_futures.append((future, x, y))
                                #print(block_futures)

                    for future, x, y in block_futures:
                        #print(future.done())
                        if future.done():
                            trans_block = future.result()
                            #block_futures.remove((future, x, y))
                            print(x,y,trans_block.shape)
                            #print(encoded_block)
                            output_frame[x:x + block_size, y:y + block_size, :] = trans_block
                        else:
                            #block_futures.remove((future, x, y))
                            block_futures.append((future, x, y))

                    #pool no!
                    for y in range(0, height, block_size):
                        for x in range(0, width, block_size):
                            #print(y,x)
                            # 提取当前块
                            block = sr[:, :, x:x + block_size, y:y + block_size]
                            #print(block.shape)
                            all_task.append(block)

                            #并行trans
                            #pool = Pool(processes=9)
                            #result = pool.apply_async(self.trans, all_task)#太慢

                    pool = Pool(16)
                    for i in range(16):
                        pool.apply_async(self.trans, all_task[i])#太慢 trans:17855 wirte:198
                    pool.close()
                    pool.join()
                    '''

                    '''2+3
                    t2 = time.time()
                    for y in range(0, height, yblock_size):
                        # 提取当前块
                        future = executor.submit(self.trans, sr[:, y:y + yblock_size, :])
                        if future.done():
                            output_frame[:, y:y + yblock_size, :] = future.result()
                        else:
                            wait([future])
                            output_frame[:, y:y + yblock_size, :] = future.result()
                    diff = time.time() - t2  ##########
                    diff2 += diff
                    self.ckp.write_log('2: {:.3f}s\n'.format(diff2))

                '''

                    diff = time.time() - t1  ##########
                    difft += diff
                    self.ckp.write_log('trans: {:.3f}s\n'.format(difft))

                    #'''
                    #t2 = time.time()  #######
                    ####vid_noblock.write(output_frame)#####################
                    ###vw.write(output_frame)
                    #diff = time.time() - t2  ##########
                    #diffw += diff
                    #self.ckp.write_log('wirte: {:.3f}s\n'.format(diffw))
                    #with open(path, "a") as f:
                    #    f.write('{:.3f}\n'.format(diff)) 
                    #'''


        #'''
                '''
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)#################################
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:##########
                        self.ckp.save_results(d, filename[0], save_list, scale)


                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)##########pnse max
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                        
                    )
                   
                )'''
        '''
        print(np.array(Frame).shape)
        t2 = time.time()  #######
        vid_noblock.write(np.array(Frame))  #####################
        diff = time.time() - t2  ##########
        diffw += diff
        self.ckp.write_log('wirte: {:.3f}s\n'.format(diffw))
        '''
        t2 = time.time()  #######
        # Encoder is asynchronous, so we need to flush it
        while True:
            success = nvEnc.FlushSinglePacket(encFrame)
            if success and (framesReceived < 30):
                byteArray = bytearray(encFrame)
                dstFile.write(byteArray)
                framesReceived += 1
                framesFlushed += 1
            else:
                break
        diff = time.time() - t2  ##########
        diffw += diff
        self.ckp.write_log('wirte: {:.3f}s\n'.format(diffw))
        print(framesReceived, "/", " 30 frames encoded and written to output file.", )
        print(framesFlushed, " frame(s) received during encoder flush.")

        #diff = time.time() - t0  ##########
        #self.ckp.write_log('\nread: {:.3f}s\n'.format(diff))
        ##out.release()#########
        dstFile.close()

        ####vid_noblock.release()#########
        ###vw.release()
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        t3 = time.time()  #######
        if self.args.save_results:
            self.ckp.end_background()
        diffs = time.time() - t3  ##########
        self.ckp.write_log('\nSave: {:.3f}s\n'.format(diffs))

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))###########
        
        self.ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)
        #'''
        torch.set_grad_enabled(True)

    # GPU function
    @cuda.jit()
    def trans_gpu(self, block):
        tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        with torch.no_grad():
            block = block[:,:, tx:tx+cuda.blockDim.x, ty:ty+cuda.blockDim.y].permute(0, 2, 3, 1).squeeze(0)
            block = block.contiguous()
            block = block.to('cpu').numpy()

    # CPU function
    #@jit()
    def trans(self, block):
        block = block.to('cpu',non_blocking=True)
        return block


    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

