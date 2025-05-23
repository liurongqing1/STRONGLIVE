import sys
from PIL import Image, ImageFilter, ImageEnhance, ImageGrab
import time
import os
import pandas as pd
import cv2
import numpy as np

#sort best patch
def get_patch(path, imgH, imgL, i, ix, iy, patch_size, scale):  #########     Top max
    ip = patch_size
    #scale=4,HR\LR patch
    retH = np.array(
        imgH[iy:iy + ip, ix:ix + ip, :]
    )  # hr patch
    #print("LR patch size:   "+str(iy / scale) + str(ix / scale))
    #print(iy, iy + ip, ix, ix + ip)
    retL = np.array(
        imgL[int(iy / scale):int(iy / scale) + int(ip / scale), int(ix / scale):int(ix / scale) + int(ip / scale), :]
    )  # lr patch

    if i + 1 < 10:
        cv2.imwrite(path + str(name) + '/sharp_patch_k/00' + str(i + 1) + '.png',
                    retH)
        cv2.imwrite(path + str(name) + '/LR_patch_k/00' + str(i + 1) + '.png', retL)
        if int(flag) == 1:
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_HR/00' + str(i + 1) + '.png',
                        retH)  ###########
            cv2.imwrite(
                '/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_LR_bicubic/X'+ str(scale) +'/00' + str(i + 1) + '.png',
                retL)  #############
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_test_LR_bicubic/X'+ str(scale) +'/00' + str(i + 1) + '.png',
                        retL)  #############
    if i + 1 < 100 and i + 1 >= 10:
        cv2.imwrite(path + str(name) + '/sharp_patch_k/0' + str(i + 1) + '.png',
                    retH)
        cv2.imwrite(path + str(name) + '/LR_patch_k/0' + str(i + 1) + '.png', retL)
        if int(flag) == 1:
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_HR/0' + str(i + 1) + '.png',
                        retH)  ###########
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_LR_bicubic/X'+  str(scale) +'/0' + str(i + 1) + '.png',
                        retL)  #############
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_test_LR_bicubic/X'+  str(scale) +'/0' + str(i + 1) + '.png',
                        retL)  #############

name = sys.argv[1]  #########cut video name Jockey_000
name2 = sys.argv[2]  #########cut video name Jockey_000
patch_size = sys.argv[3]  ###cut size 400
scale = sys.argv[4]  ###downsample scale 4
flag = sys.argv[5]  ########## flag = 1
path = '/data/lrq/edge_dection/'

C = []
ip = int(patch_size)
for ix in range(0, 1200 - ip + 1, ip):
    for iy in range(0, 1200 - ip + 1, ip):
        C.append([ix, iy])
print(C)
imgH = cv2.imread(path+str(name)+'/LR_bicubic/X'+  str(scale) +'/1.png')
#print(imgH.size)
#imgH = cv2.imread('H:/edge_dection/Ready_008/009.png')
imgL = cv2.imread(path+str(name)+'/LR_bicubic/X'+  str(scale) +'/2.png')
#imgL = cv2.imread('H:/edge_dection/Ready_008/009lr.png')
print(imgL.shape)

if (not os.path.exists(path + str(name) + '/sharp_patch_k/')):
    os.makedirs(path + str(name) + '/sharp_patch_k/')
if (not os.path.exists(path + str(name) + '/LR_patch_k/')):
    os.makedirs(path + str(name) + '/LR_patch_k/')

for i in range(len(C)):###########10 patch ；i,j变
    ix = C[i][0]
    iy = C[i][1]
    get_patch(path, imgH, imgL, i, ix, iy, int(patch_size), int(scale))

if name2=='Ready_009':
    imgH = cv2.imread(path + str(name) + '/LR_bicubic/X' + str(scale) + '/11.png')
    # print(imgH.size)
    # imgH = cv2.imread('H:/edge_dection/Ready_008/009.png')
    imgL = cv2.imread(path + str(name) + '/LR_bicubic/X' + str(scale) + '/22.png')
    # imgL = cv2.imread('H:/edge_dection/Ready_008/009lr.png')
    print(imgL.shape)

    if (not os.path.exists(path + str(name) + '/sharp_patch_k/')):
        os.makedirs(path + str(name) + '/sharp_patch_k/')
    if (not os.path.exists(path + str(name) + '/LR_patch_k/')):
        os.makedirs(path + str(name) + '/LR_patch_k/')


    for i in range(9,10+len(C)-1):  ###########18 patch ；i,j变 9-17
        #print(i)
        ix = C[i-10][0]
        iy = C[i-10][1]
        get_patch(path, imgH, imgL, i, ix, iy, int(patch_size), int(scale))

#①pre-model other test + onlie-model other test result
#os.system("bash /data/lrq/edge_dection/other_test.sh %s %s %s %s" % (name,name2,scale,i+1))