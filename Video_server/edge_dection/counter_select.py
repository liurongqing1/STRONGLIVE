import sys
from PIL import Image, ImageFilter, ImageEnhance, ImageGrab
import time
import os
import pandas as pd
import cv2
import numpy as np

os.environ['MKL_THREADING_LAYER'] = 'GNU'
#pre ready Jockey_000/HR/001.png

#sharpness
def imageResize(input_path, output_path):
    # 获取输入文件夹中的所有文件/夹，并改变工作空间
    files = os.listdir(input_path)
    #print(input_path)
    os.chdir(input_path)
    # 判断输出文件夹是否存在，不存在则创建
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    for file in files:
        # 判断是否为文件，文件夹不操作
        if (os.path.isfile(file)):
            imgH = Image.open(file)
            #print(img.size)
            # 增强锐度
            sharpness = ImageEnhance.Sharpness(imgH)
            img = sharpness.enhance(10)
            #img.save(os.path.join(output_path, file))
            return img,imgH


def get_loc(img, patch_size):  #########随机补丁位置
    ih, iw = img.shape[:2]
    print("分辨率：" + str(ih) + "x" + str(iw))
    # print(ih,iw)
    ip = patch_size
    C = []
    R = []
    L = []
    for ix in range(0, iw - ip + 1, 200):
        for iy in range(0, ih - ip + 1, 200):
            # print(ix,iy)
            ret = np.array(
                img[iy:iy + ip, ix:ix + ip]
            )  # 隔200取一个，分块
            # print(ret.shape)
            # cv2.imwrite('H:/edge_dection/Ready_008/'+str(ix)+"_"+str(iy)+'.png',ret )

            contours, hierarchy = cv2.findContours(ret, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找检测物体的轮廓
            R.append([ix, iy])  # 保存每个块起始位置
            # print(len(contours))
            C.append(len(contours))  # 保存每个块轮廓数

    '''取最大轮廓数索引，找最大轮廓位置，保存前10个块'''
    print("总块数：", len(C))
    num = len(C)
    print("提取块数：", int(len(C)))
    for i in range(int(len(C))):  ###########10 patch
        print("最大轮廓数：", max(C))  ####
        index = C.index(max(C))  #####
        print("最大值索引：", index)
        print("位置：      ", R[index])
        L.append(R[index])
        del C[index]
        del R[index]
    return num, L



#sort best patch
def get_patch(path, imgH, imgL, i, ix, iy, patch_size, scale):  #########     Top max
    ip = patch_size
    #scale=4,HR\LR patch
    retH = np.array(
        imgH[iy:iy + ip, ix:ix + ip, :]
    )  # hr patch
    print("LR patch size:   "+str(iy / scale) + str(ix / scale))
    print(iy, iy + ip, ix, ix + ip)
    retL = np.array(
        imgL[int(iy / scale):int(iy / scale) + int(ip / scale), int(ix / scale):int(ix / scale) + int(ip / scale), :]
    )  # lr patch

    if i + 1 < 10:
        cv2.imwrite(path + str(name) + '/sharp_patch_l/00' + str(i + 1) + '.png',
                    retH)
        cv2.imwrite(path + str(name) + '/LR_patch_l/00' + str(i + 1) + '.png', retL)
        if int(flag) == 1:
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_HR/00' + str(i + 1) + '.png',
                        retH)  ###########
            cv2.imwrite(
                '/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_LR_bicubic/X'+ str(scale) +'/00' + str(i + 1) + '.png',
                retL)  #############
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_test_LR_bicubic/X'+ str(scale) +'/00' + str(i + 1) + '.png',
                        retL)  #############

    if i + 1 < 100 and i + 1 >= 10:
        cv2.imwrite(path + str(name) + '/sharp_patch_l/0' + str(i + 1) + '.png',
                    retH)
        cv2.imwrite(path + str(name) + '/LR_patch_l/0' + str(i + 1) + '.png', retL)
        if int(flag) == 1:
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_HR/0' + str(i + 1) + '.png',
                        retH)  ###########
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_LR_bicubic/X'+  str(scale) +'/0' + str(i + 1) + '.png',
                        retL)  #############
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_test_LR_bicubic/X'+  str(scale) +'/0' + str(i + 1) + '.png',
                        retL)  #############
    if i + 1 >= 100:
        cv2.imwrite(path + str(name) + '/sharp_patch_l/' + str(i + 1) + '.png',
                    retH)
        cv2.imwrite(path + str(name) + '/LR_patch_l/' + str(i + 1) + '.png', retL)
        if int(flag) == 1:
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_HR/' + str(i + 1) + '.png',
                        retH)  ###########
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_LR_bicubic/X'+  str(scale) +'/' + str(i + 1) + '.png',
                        retL)  #############
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_test_LR_bicubic/X'+  str(scale) +'/' + str(i + 1) + '.png',
                        retL)  #############


#vmaf collect
def read_vmafxls(path, name):
    df = pd.read_csv(path + str(name) +"/"+ str(name) + '_vmaf.xls')
    V = list(df["vmaf"])
    print("———————————————Finish Best Patch VMAF———————————————")
    return V





name = sys.argv[1]  #########cut video name Jockey_000
name2 = sys.argv[2]  #########cut video name Ready_000
patch_size = int(sys.argv[3])  ###cut size 400
scale = sys.argv[4]  ###downsample scale 4
flag = sys.argv[5]  ########## flag = 1
h = 3840
w = 2160
path = '/data/lrq/edge_dection/'
#  源目录
input_path = '/data/lrq/edge_dection/'+str(name)+'/HR/'
#  输出目录
output_path = '/data/lrq/edge_dection/'+str(name)+'/sharp/'

'''
'''
img,imgH =imageResize(input_path, output_path) #1.sharpness
imgH = np.asarray(imgH.convert('L'))
imgH= cv2.Canny(imgH,10,100)

num,L= get_loc(np.array(imgH),patch_size)#10个块位置
#裁切位置
#save_loc(path,name,L)

img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
print(img.size)
imgL = cv2.imread('/home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/' + str(name) + '/LR_bicubic/X'+ scale +'/001.png')
print('LR shape:  '+str(imgL.shape))
if (not os.path.exists(path + str(name) + '/sharp_patch_l/')):
    os.makedirs(path + str(name) + '/sharp_patch_l/')
if (not os.path.exists(path + str(name) + '/LR_patch_l/')):
    os.makedirs(path + str(name) + '/LR_patch_l/')
for i in range(num):  ###########10 patch   i,j
    ix = L[i][0]
    iy = L[i][1]
    get_patch(path, img, imgL, i, ix, iy, int(patch_size), int(scale)) #6.
print("———————————————Finish Best Patch———————————————")

#os.system("bash /data/lrq/edge_dection/patch_train.sh %s %s" % (name,scale)) #7.
'''
M = read_vmafxls(path, name)
print("VMAF:   ", M)
print("max VMAF: ", max(M))  ####
index = M.index(max(M))  #####
print("max index:  ", index)
best_num = index*20+1   #8.
print("best_epoch:  ", best_num)
'''
best_num=10##########
#①pre-model other test + onlie-model other test result
os.system("bash /data/lrq/edge_dection/other_test.sh %s %s %s %s" % (name,name2,scale,best_num))
#②onlie-model self test result
os.system("bash /data/lrq/edge_dection/self_test.sh %s %s %s" % (name,scale,best_num))
print("———————————————Finish Retrain and test———————————————")
