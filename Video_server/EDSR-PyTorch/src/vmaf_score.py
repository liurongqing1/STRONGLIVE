import sys
import os
import cv2 as cv

#src = cv.imread("test.jpg")
def trans(png):
    yuv = cv.cvtColor(pmg, cv.COLOR_BGR2YUV)
    return yuv

#'''
def png_yuv(sr,hr):
    index = 0
    sr = sr.transpose(1,3)#[16,3,40,40]->[16,40,40,3]
    hr = sr.transpose(1,3)
    for i in range(len(hr)):
        print("num:",str(i))
        sr_yuv = trans(sr[i])
        hr_yuv = trans(hr[i]) 
        cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/yuv/sr/'+str(i)+'.yuv', sr_yuv)
        cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/yuv/hr/'+str(i)+'.yuv', hr_yuv)
    return len(hr)
        
#'''

#'''
def VMAF(sr,hr):
    bs = png_yuv(sr, hr)
    sys.path.append(sys.path.insert(0, '/home/srteam/lrq/vmaf/python'))
    print(sys.path[0])
    from vmaf.script.run_vmaf import VMAF
    LOSS = []
    for i in reage(bs):
        VMAF_socre=VMAF('/home/srteam/lrq/EDSR-PyTorch/yuv/sr/'+str(i)+'.yuv','/home/srteam/lrq/EDSR-PyTorch/yuv/hr/'+str(i)+'.yuv')
        loss = (100.-float(VMAF_socre))/100.
        #print('VMAF:',VMAF_socre,"\nloss:",str(loss))
        LOSS.append(loss)
    l=sum(LOSS)/len(LOSS)
    print("\nloss:", str(l))
    return l
'''
sys.path.append(sys.path.insert(0, '/home/srteam/lrq/vmaf/python'))
print(sys.path[0])
from vmaf.script.run_vmaf import VMAF
VMAF_socre=VMAF('/home/srteam/lrq/vmaf/1.yuv ','/home/srteam/lrq/vmaf/2.yuv')
print('VMAF:',VMAF_socre)
'''
