import numpy as np
from PIL import Image
import cv2
import math
import cv2
import argparse
import glob
import time
import os
import core
import torch
from torch import cuda
from torchvision import transforms
from  torchvision import utils as vutils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

transform = transforms.Compose([transforms.ToTensor()])

def Bicubic(skip, scale, name):
    diffr=0
    diffb=0
    diffs=0
    dir = '/home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/{}/LR_bicubic/X{}'.format(name, str(scale))
    hr = sorted(glob.glob(dir+'/*.png'))
    #'*' + ".png"
    print("hr:" + str(dir))  ######

    for i in range(1, len(hr) + 1, int(skip) + 1):  ######
        print("skip:" + str(i))  ######
        if i <10:
            path = os.path.join('/home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/{}/LR_bicubic/X{}/00{}{}'.format(
                name, str(scale), str(i), ".png"))
        if i >=10:
            path = os.path.join('/home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/{}/LR_bicubic/X{}/0{}{}'.format(
                name, str(scale), str(i), ".png"))
        print(path)
        #img = np.array((Image.open(path)))

        t1 = time.time()  #########read
        img = cv2.imread(path)
        img = transform(img)
        diff = time.time() - t1  ########
        diffr += diff
        print("read:" + str(diffr))

        #print(img.size())

        if cuda.is_available():#######max time!!!
            img = img.cuda()

        #bicubic
        t0 = time.time()  ########bicubic
        #print(img.shape) #torch.Size([3, 540, 960])
        new_img = core.imresize(img, scale)
        #print(new_img)
        diff = time.time() - t0########
        #print("bicubic_per:"+str(diff))
        diffb += diff
        print("bicubic:"+str(diffb))

        with open('/home/srteam/lrq/EDSR-PyTorch/experiment/{}/bicubic{}.txt'.format(name, scale), "a") as f:
            f.write('{:.3f}\n'.format(diff))
        #save
        #print(new_img.size())
        if i <10:
            path_sr = os.path.join('/home/srteam/lrq/EDSR-PyTorch/experiment/{}/results-{}/X{}/30/00{}{}'.format(
                name, name, str(scale), str(i) ,".png"))
        if i >=10:
            path_sr=os.path.join('/home/srteam/lrq/EDSR-PyTorch/experiment/{}/results-{}/X{}/30/0{}{}'.format(
                name, name, str(scale), str(i),".png"))
        #print("path:" + str(path_sr))
        #toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
        #pic = toPIL(new_img)
        #pic.save(path_sr)

        t2 = time.time()  ########save
        vutils.save_image(new_img, path_sr, normalize=True)
        diff = time.time() - t2 ########
        diffs += diff
        print("save:"+str(diffs))

    return diffs


parser = argparse.ArgumentParser()
parser.add_argument('--skip', type=int, default=0, help='skip')
parser.add_argument('--scale', type=int, default=4, help='super resolution scale')
parser.add_argument('--name', type=str, default='Bosp', help='test dataset name')
args = parser.parse_args()
diffs = Bicubic(args.skip, args.scale, args.name)
print("bicubic:"+str(diffb))
'''
skip=0
diffs=0
hr = sorted(
    glob.glob(
        os.path.join('/home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Ready/HR',
                         '*' + ".png")))

for i in range(1, len(hr) + 1, int(skip) + 1):  ######
    print("skip:" + str(i))  ######
    path = os.path.join('/home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Ready/HR',
                          str(i) + ".png")

    t0 = time.time()  ########
    img = cv2.imread(path)
    diff = time.time() - t0########

    diffs += diff
    print("diffs:"+str(diffs))
'''