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



class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.skip = args.skip
        self.ckp = ckp
        self.loader_train = loader.loader_train  # data.Data
        self.loader_test = loader.loader_test

        self.model = my_model
        self.loss = my_loss
        # 待训练参数
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
        # '''冻结
        para_name = ['model.head.weight', 'model.head.bias',

                     'model.warmup.0.weight', 'model.warmup.0.bias',
                     'model.warmup.1.bn1.weight', 'model.warmup.1.bn1.bias', 'model.warmup.1.bn2.weight',
                     'model.warmup.1.bn2.bias', 'model.warmup.1.bn3.weight', 'model.warmup.1.bn3.bias',
                     'model.warmup.1.squeeze.weight', 'model.warmup.1.squeeze.bias',
                     'model.warmup.1.convs.0.conv1.expand_conv.weight',
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
                     'model.warmup.1.convs.1.conv2.reduce_conv.bias', 'model.warmup.1.excitate.weight',
                     'model.warmup.1.excitate.bias',

                     'model.ERBs.0.conv1.expand_conv.weight', 'model.ERBs.0.conv1.expand_conv.bias',
                     'model.ERBs.0.conv1.fea_conv.weight',
                     'model.ERBs.0.conv1.fea_conv.bias', 'model.ERBs.0.conv1.reduce_conv.weight',
                     'model.ERBs.0.conv1.reduce_conv.bias',
                     'model.ERBs.0.conv2.expand_conv.weight', 'model.ERBs.0.conv2.expand_conv.bias',
                     'model.ERBs.0.conv2.fea_conv.weight',
                     'model.ERBs.0.conv2.fea_conv.bias', 'model.ERBs.0.conv2.reduce_conv.weight',
                     'model.ERBs.0.conv2.reduce_conv.bias',

                     'model.ERBs.1.conv1.expand_conv.weight', 'model.ERBs.1.conv1.expand_conv.bias',
                     'model.ERBs.1.conv1.fea_conv.weight',
                     'model.ERBs.1.conv1.fea_conv.bias', 'model.ERBs.1.conv1.reduce_conv.weight',
                     'model.ERBs.1.conv1.reduce_conv.bias',
                     'model.ERBs.1.conv2.expand_conv.weight', 'model.ERBs.1.conv2.expand_conv.bias',
                     'model.ERBs.1.conv2.fea_conv.weight',
                     'model.ERBs.1.conv2.fea_conv.bias', 'model.ERBs.1.conv2.reduce_conv.weight',
                     'model.ERBs.1.conv2.reduce_conv.bias',

                     'model.ERBs.2.conv1.expand_conv.weight', 'model.ERBs.2.conv1.expand_conv.bias',
                     'model.ERBs.2.conv1.fea_conv.weight', 'model.ERBs.2.conv1.fea_conv.bias',
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

                     ]  #
        for name, parameter in self.model.named_parameters():
            # self.ckp.write_log("'{}',".format(name))
            if name in para_name:
                # self.ckp.write_log("{}".format(parameter))
                parameter.requires_grad = False
                # self.ckp.write_log("{}".format(str(i)))
            self.ckp.write_log('{}: {}'.format(name, parameter.requires_grad))
        # '''
        # 优化器
        self.optimizer = utility.make_optimizer(args, self.model)  #############

        if self.args.load != '':
            self.optimizer.load(ckp.dir,
                                epoch=len(ckp.log))  # ckp.dir='/home/srteam/lrq/EDSR-PyTorch', 'experiment', args.load

        self.error_last = 1e8

    def train(self):
        t0 = time.time()  #######
        self.loss.step()  ######lr_scheduler.step
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)  ##
        for batch, (lr, hr, _,) in enumerate(self.loader_train):  # [6, 3, 192, 192]
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)  ##############
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()  # 更新参数
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:  ##################
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
        t1 = time.time()  #######
        train_t = t1 - t0
        self.ckp.write_log('Train time：{:.3f}s\n'.format(train_t))  ######

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()  ##########

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )

        self.model.eval()

        timer_test = utility.timer()
        diffm = 0
        diffg = 0
        difft = 0
        diffw = 0

        num = -1
        framesReceived = 0
        framesFlushed = 0
        pkt = []
        N_bic = []

        encFilePath = "/home/srteam/lrq/EDSR-PyTorch/gpucode/Jockey_000/Jockey_000540.mp4"#
        nvDec = nvc.PyNvDecoder(encFilePath, 0)
        # Convert to planar RGB
        to_rgb = SamplePyTorch.cconverter(960, 540, 0)
        to_rgb.add(nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420)
        to_rgb.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB)
        to_rgb.add(nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR)

        dstFile = open('/home/srteam/lrq/EDSR-PyTorch/gpucode/Jockey_000/result.mp4', "wb")
        nvEnc = nvc.PyNvEncoder(
            {"codec": "h264", 's': '3840x2160', 'preset': 'P4', "profile": "high", "tuning_info": "high_quality",
             "bitrate": "10M", "fps": "30", "gop": str(self.skip)}, 0,
            nvc.PixelFormat.NV12)  #
        nvEnc2 = nvc.PyNvEncoder(
            {"codec": "h264", 's': '3840x2160', 'preset': 'P4', "profile": "high", "tuning_info": "high_quality",
             "bitrate": "20M", "fps": "30", "gop": "1"}, 0,
            nvc.PixelFormat.NV12)  #
        # Convert back to NV12
        to_nv12 = SamplePyTorch.cconverter(3840, 2160, 0)  ####
        to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
        to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
        to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)
        encFrame = np.ndarray(shape=(0), dtype=np.uint8)  ###
        encFrame2 = np.ndarray(shape=(0), dtype=np.uint8)  ###

        #path = r'/home/srteam/lrq/EDSR-PyTorch/experiment/{}/test{}.txt'.format(self.args.data_test[0], self.scale[0])

        # t0 = time.time() #resd start
        for idx_scale, scale in enumerate(self.scale):  # 0,4
            '''
            ####################ffmpecv decode###############
            vid_noblock = ffmpegcv.noblock(ffmpegcv.VideoCapture, encFilePath, pix_fmt='rgb24')
            for _ in tqdm.trange(30):
                t4 = time.time()  #######
                ret, frame = vid_noblock.read()  # 当前图像已经被缓冲，不会占用时间
                with torch.no_grad():
                    frame = torch.from_numpy(frame).pin_memory()
                    Frame = frame.to('cuda')
                    Frame = Frame.to(dtype = torch.float32)
                    Frame = Frame.permute(2, 0, 1)
                    lr = Frame.unsqueeze(0)
                    #Frame = Frame.view(1, 540, 960, 3)
                    #lr = Frame.permute(0, 3, 1, 2)
            '''
            # decode
            while True:
                t4 = time.time()  #######
                src_surface = nvDec.DecodeSingleSurface()
                if src_surface.Empty():
                    break
                rgb_pln = to_rgb.run(src_surface)
                if rgb_pln.Empty():
                    break
                src_tensor = SamplePyTorch.surface_to_tensor(rgb_pln)  #############
                #src_tensor = src_tensor[[2, 1, 0], :, :] #whem save png need!!


                '''
                array =  src_tensor.cpu().numpy()
                # 将数组从通道x高x宽的顺序转换为高x宽x通道的顺序
                array = np.transpose(array, (1, 2, 0))
                # 将通道顺序从 (R, G, B) 转换为 (B, G, R)
                #array = array[:, :, ::-1]
                self.ckp.write_log('decode: {}s\n'.format(array))
                # 使用OpenCV保存图像
                cv2.imwrite("/home/srteam/lrq/EDSR-PyTorch/output.png", array)  # 根据需要更改文件名和格式
                image = cv2.imread('/home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Jockey_000/LR_bicubic/X4/030.png', cv2.IMREAD_UNCHANGED)
                #self.ckp.write_log('decode: {}s\n'.format(image))
                '''

                lr = src_tensor.unsqueeze(0)
                #self.ckp.write_log('tensor: {}\n'.format(lr))

                diff = time.time() - t4  ##########
                diffg += diff
                self.ckp.write_log('decode: {:.3f}s\n'.format(diffg))

                #self.ckp.write_log('lr: {}\n'.format(lr))
                num += 1
                t0 = time.time()  #######
                # print(lr.shape) #torch.Size([1, 3, 540, 960])
                if num == 0:
                    sr = self.model(lr, idx_scale)  # tensor().cuda (1,3,2160,3840)
                    self.ckp.write_log('sr num is :{:01d}\n'.format(num))
                if num < 30 and num % self.skip == 0:
                    sr = self.model(lr, idx_scale)  # tensor().cuda (1,3,2160,3840)
                    self.ckp.write_log('sr num is :{:01d}\n'.format(num))

                bic = core.imresize(lr.squeeze(0), self.scale[0])
                self.ckp.write_log('bicubic num is :{:01d}\n'.format(num))
                diff = time.time() - t0
                diffm += diff
                self.ckp.write_log('SR: {:.3f}s\n'.format(diffm))


                gpu_id = 0  # GPU ID
                t1 = time.time()  #######
                ################异步插帧.3###############

                encFrame2 = np.ndarray(shape=(0), dtype=np.uint8)  ###
                if num == 0:
                    sr = sr.squeeze(0).contiguous()  ######torch.Size([3, 2160, 3840]
                    sr_surface = to_nv12.run(SamplePyTorch.tensor_to_surface(sr, gpu_id))
                    # Encoded video frame
                    success_sr = nvEnc2.EncodeSingleSurface(sr_surface, encFrame2, sync=True)  # False，则编码操作是异步的

                if (num - 3) % self.skip == 0:
                    sr = sr.squeeze(0).contiguous()  ######torch.Size([3, 2160, 3840]
                    sr_surface = to_nv12.run(SamplePyTorch.tensor_to_surface(sr, gpu_id))
                    # Encoded video frame
                    success_sr = nvEnc2.EncodeSingleSurface(sr_surface, encFrame2, sync=True)  # False，则编码操作是异步的

                # encFrame = np.ndarray(shape=(0), dtype=np.uint8)  ###
                bic = bic.squeeze(0).contiguous()
                bic_surface = to_nv12.run(SamplePyTorch.tensor_to_surface(bic, gpu_id))
                success_bic = nvEnc.EncodeSingleSurface(bic_surface, encFrame, sync=False)  # False，则编码操作是异步的

                if num == 0:
                    if success_sr:
                        pkt.append(encFrame2)  # .copy()
                        framesReceived += 1
                if num > 3 and (num - 3) % self.skip == 0:
                    if success_sr:
                        pkt.append(encFrame2)  # .copy()
                        framesReceived += 1

                if success_bic:
                    if num > 3 and (num - 3) % self.skip != 0 and num != 3:
                        pkt.append(encFrame.copy())
                        framesReceived += 1
                else:
                    N_bic.append(num)

                diff = time.time() - t1  ##########
                difft += diff
                self.ckp.write_log('trans: {:.3f}s\n'.format(difft))


        t2 = time.time()  #######
        while True:
            success = nvEnc.FlushSinglePacket(encFrame)
            if success:
                f = encFrame.copy()
                pkt.append(f)
                framesReceived += 1
                framesFlushed += 1
                print("\nbic flush " + str(f) + '\n')
            else:
                break
        for pkt_frame in pkt:
            dstFile.write(bytearray(pkt_frame))

        diff = time.time() - t2  ##########
        diffw += diff
        self.ckp.write_log('wirte: {:.3f}s\n'.format(diffw))
        # '''
        print(framesReceived, "/", " 30 frames encoded and written to output file.", )
        print(framesFlushed, " frame(s) received during encoder flush.")

        dstFile.close()
        
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))  ###########

        self.ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)
        # '''
        torch.set_grad_enabled(True)


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
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs