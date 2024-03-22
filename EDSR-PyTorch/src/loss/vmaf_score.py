import sys
import os
import cv2 as cv
import torch.nn as nn
import torch
import cv2
from ffmpy3 import FFmpeg

# src = cv.imread("test.jpg")

#'''

def tensortoyuv(input_tensor):#filename
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    #print(len(input_tensor.shape),input_tensor.shape[0] )
    assert (len(input_tensor.shape) == 3 and input_tensor.shape[0] == 3)
    # 复制一份
    #input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    #input_tensor = input_tensor.squeeze()
    #print("1:",input_tensor.shape,input_tensor.type)# torch.Size([3, 400, 400]) tensor
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.permute(1, 2, 0).type(torch.uint8).numpy()
    #print("2:",input_tensor.shape)#(400, 400, 3) numpy.ndarray
    # RGB转BRG
    #input_tensor = cv2.cvtColor(input_tensor, cv.COLOR_BGR2YCrCb)#cv.COLOR_BGR2YUV
    #input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2YUV_I420)
    #print("3:",input_tensor.shape)#(400, 400, 3)
    #cv2.imwrite(filename, input_tensor) numpy.ndarray
    return input_tensor

def png_yuv(sr, hr):
    #sr = sr.transpose(1, 3)  # [16,3,40,40]->[16,40,40,3]
    #hr = sr.transpose(1, 3)
    #print(sr.shape,sr.type,len(sr))
    #torch.Size([16, 400, 400, 3]) <built-in method type of Tensor object at 0x7f734fcb54a0> 16
    fps = 1
    for i in range(len(hr)):
        print("num:", str(i))
        sryuv = tensortoyuv(sr[i])#png->yuv
        hryuv = tensortoyuv(hr[i])
        #print(sr_yuv.shape[0])
        size = '{}x{}'.format(sryuv.shape[1], sryuv.shape[0])
        index_yuv = str(i) + '.yuv'
        index_png = str(i) + '.png'
        sr_png = os.path.join('/home/srteam/lrq/EDSR-PyTorch/yuv/sr/', index_png )
        hr_png = os.path.join('/home/srteam/lrq/EDSR-PyTorch/yuv/hr/', index_png )
        sr_yuv = os.path.join('/home/srteam/lrq/EDSR-PyTorch/yuv/sr/', index_yuv )
        hr_yuv = os.path.join('/home/srteam/lrq/EDSR-PyTorch/yuv/hr/', index_yuv )

        cv2.imwrite(sr_png, sryuv)
        cv2.imwrite(hr_png, hryuv)

        ff = FFmpeg(inputs={sr_png: None},outputs={sr_yuv: '-y -s {} -pix_fmt yuv420p'.format(size)})
        #print(ff.cmd)
        ff.run()
        ff = FFmpeg(inputs={hr_png: None},outputs={hr_yuv: '-y -s {} -pix_fmt yuv420p'.format(size)})
        #print(ff.cmd)
        ff.run()

        # ffmpeg_quality_metrics /home/srteam/lrq/EDSR-PyTorch/yuv/sr/1.yuv /home/srteam/lrq/EDSR-PyTorch/yuv/hr/1.yuv - m vmaf --model-path /home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl

        #fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
        #videoWriter1 = cv2.VideoWriter('/home/srteam/lrq/EDSR-PyTorch/yuv/sr/'+ index, fourcc, fps, (sr_yuv.shape[0], sr_yuv.shape[1]))
        #videoWriter1.write(sr_yuv)
        #videoWriter2 = cv2.VideoWriter('/home/srteam/lrq/EDSR-PyTorch/yuv/hr/'+ index, fourcc, fps, (sr_yuv.shape[0], sr_yuv.shape[1]))
        #videoWriter2.write(hr_yuv)
    #videoWriter1.release()
    #videoWriter2.release()
        #cv2.imwrite(os.path.join('/home/srteam/lrq/EDSR-PyTorch/yuv/sr/', index ), sr_yuv)
        #cv2.imwrite(os.path.join('/home/srteam/lrq/EDSR-PyTorch/yuv/hr/', index ), hr_yuv)
    return len(hr)


#'''

# '''
class VMAF(nn.Module):
    def __init__(self,srgs):
        super(VMAF, self).__init__()
    '''
        def trans(png):
            yuv = cv.cvtColor(png, cv.COLOR_BGR2YUV)
            return yuv

        def png_yuv(sr, hr):
            index = 0
            sr = sr.transpose(1, 3)  # [16,3,40,40]->[16,40,40,3]
            hr = sr.transpose(1, 3)
            for i in range(len(hr)):
                print("num:", str(i))
                sr_yuv = trans(sr[i])
                hr_yuv = trans(hr[i])
                cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/yuv/sr/' + str(i) + '.yuv', sr_yuv)
                cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/yuv/hr/' + str(i) + '.yuv', hr_yuv)
            return len(hr)
        '''

    def forward(self, sr, hr):
        bs = png_yuv(sr, hr)
        sys.path.append(sys.path.insert(0, '/home/srteam/lrq/vmaf/python'))
        print(sys.path[0])
        from vmaf.script.run_vmaf import VMAF
        LOSS = []
        for i in range(bs):
            VMAF_socre = VMAF('/home/srteam/lrq/EDSR-PyTorch/yuv/sr/' + str(i) + '.yuv',
                            '/home/srteam/lrq/EDSR-PyTorch/yuv/hr/' + str(i) + '.yuv')
            loss = (100. - float(VMAF_socre)) / 100.
            # print('VMAF:',VMAF_socre,"\nloss:",str(loss))
            LOSS.append(loss)
        l = sum(LOSS) / len(LOSS)
        l = torch.Tensor([l]).cuda()#torch.FloatTensor(l)
        #print("\nloss:", str(l))
        return l


'''
sys.path.append(sys.path.insert(0, '/home/srteam/lrq/vmaf/python'))
print(sys.path[0])
from vmaf.script.run_vmaf import VMAF
VMAF_socre=VMAF('/home/srteam/lrq/vmaf/1.yuv ','/home/srteam/lrq/vmaf/2.yuv')
print('VMAF:',VMAF_socre)
'''
