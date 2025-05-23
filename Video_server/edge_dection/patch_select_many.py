import sys
from PIL import Image, ImageFilter, ImageEnhance, ImageGrab
import time
import os
import pandas as pd
import cv2
import numpy as np

os.environ['MKL_THREADING_LAYER'] = 'GNU'


# pre ready Jockey_000/HR/001.png

# sharpness
def imageResize(input_path, output_path):
    files = os.listdir(input_path)
    os.chdir(input_path)
    # print(input_path)
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    for file in files:
        if (os.path.isfile(file)):
            img = Image.open(file)
            sharpness = ImageEnhance.Sharpness(img)
            img = sharpness.enhance(10)
            img.save(os.path.join(output_path, file))
            print("———————————————Finish Sharpness———————————————")


# blur
class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


def blur_patch(path, ip, h, w):
    C = []
    for ix in range(1, h - ip + 1, ip - 200):
        for iy in range(1, w - ip + 1, ip - 200):
            C.append([ix - 1, iy - 1])
            bounds = (ix, iy, ix + ip, iy + ip)
            image = Image.open(path + str(name) + '/sharp/001.png')
            image = image.filter(MyGaussianBlur(radius=29, bounds=bounds))
            output_path = path + str(name) + '/blur/'
            if (not os.path.exists(output_path)):
                os.makedirs(output_path)
            image.save(output_path + str(ix) + '-' + str(iy) + '.png')
    print(len(C))
    print("———————————————Finish Cut sharrpnes img———————————————")
    return C


# vmaf collect
def read_xls(path, name):
    df = pd.read_csv(path + str(name) + '/vmaf' + str(scale) + '.xls')
    V = list(df["vmaf"])
    print("———————————————Finish Blur Patch VMAF———————————————")
    return V


# vmaf min location get
def get_loc(V, C):
    L = []
    print("patch sum： ", len(V))
    print("extect patch sum： ", int(len(V)))
    for i in range(int(len(V))):  ###########10 patch
        print("min VMAF: ", min(V))  ####
        index = V.index(min(V))  #####
        print("min index:  ", index)
        print("location:  ", C[index])
        L.append(C[index])
        del V[index]
        del C[index]
    print("———————————————Finish Min Vmaf Location———————————————")
    return L


# vmaf min location save
def save_loc(path, name, L):
    # list loc dataframe
    df = pd.DataFrame(L, columns=list('xy'))
    # save vmaf location to excel, best patch location
    df.to_excel(path + str(name) + '/' + str(name) + "_vmaf_loc" + str(scale) + ".xlsx", index=False)
    print("———————————————Finish Save Location———————————————")


# sort best patch
def get_patch(path, imgH, imgL, i, ix, iy, patch_size, scale):  #########     Top max
    ip = patch_size
    # scale=4,HR\LR patch
    retH = np.array(
        imgH[iy:iy + ip, ix:ix + ip, :]
    )  # hr patch
    # print("LR patch size:   "+str(iy / scale) + str(ix / scale))
    # print(iy, iy + ip, ix, ix + ip)
    retL = np.array(
        imgL[int(iy / scale):int(iy / scale) + int(ip / scale), int(ix / scale):int(ix / scale) + int(ip / scale), :]
    )  # lr patch

    if i + 1 < 10:
        cv2.imwrite(path + str(name) + '/sharp_patch/00' + str(i + 1) + '.png',
                    retH)
        cv2.imwrite(path + str(name) + '/LR_patch/00' + str(i + 1) + '.png', retL)
        if int(flag) == 1:
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_HR/00' + str(i + 1) + '.png',
                        retH)  ###########
            cv2.imwrite(
                '/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_LR_bicubic/X' + str(scale) + '/00' + str(
                    i + 1) + '.png',
                retL)  #############
            cv2.imwrite(
                '/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_test_LR_bicubic/X' + str(scale) + '/00' + str(
                    i + 1) + '.png',
                retL)  #############

    if i + 1 < 100 and i + 1 >= 10:
        cv2.imwrite(path + str(name) + '/sharp_patch/0' + str(i + 1) + '.png',
                    retH)
        cv2.imwrite(path + str(name) + '/LR_patch/0' + str(i + 1) + '.png', retL)
        if int(flag) == 1:
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_HR/0' + str(i + 1) + '.png',
                        retH)  ###########
            cv2.imwrite(
                '/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_LR_bicubic/X' + str(scale) + '/0' + str(
                    i + 1) + '.png',
                retL)  #############
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_test_LR_bicubic/X' + str(scale) + '/0' + str(
                i + 1) + '.png',
                        retL)  #############
    if i + 1 >= 100:
        cv2.imwrite(path + str(name) + '/sharp_patch/' + str(i + 1) + '.png',
                    retH)
        cv2.imwrite(path + str(name) + '/LR_patch/' + str(i + 1) + '.png', retL)
        if int(flag) == 1:
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_HR/' + str(i + 1) + '.png',
                        retH)  ###########
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_train_LR_bicubic/X' + str(scale) + '/' + str(
                i + 1) + '.png',
                        retL)  #############
            cv2.imwrite('/home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/DIV2K_test_LR_bicubic/X' + str(scale) + '/' + str(
                i + 1) + '.png',
                        retL)  #############


# vmaf collect
def read_vmafxls(path, name):
    df = pd.read_csv(path + str(name) + "/" + str(name) + '_vmaf' + str(scale) + '.xls')  #############3
    V = list(df["vmaf"])
    print("———————————————Finish Best Patch VMAF———————————————")
    return V


# python /data/lrq/edge_dection/patch_select.py Jockey_000 Jockey_001 400 4 1


name = sys.argv[1]  #########cut video name Jockey_000
patch_size = sys.argv[2]  ###cut size 400
scale = sys.argv[3]  ###downsample scale 4
flag = sys.argv[4]  ########## flag = 1
h = 3840
w = 2160
path = '/data/lrq/edge_dection/'
#  源目录
input_path = '/data/lrq/edge_dection/' + str(name) + '/HR/'
#  输出目录
output_path = '/data/lrq/edge_dection/' + str(name) + '/sharp/'

'''

imageResize(input_path, output_path) #1.sharpness

C = blur_patch(path, int(patch_size), h, w) #2.blur
'''
C = []
ip = int(patch_size)
for ix in range(1, 3840 - ip + 1, ip - 200):
    for iy in range(1, 2160 - ip + 1, ip - 200):
        C.append([ix - 1, iy - 1])

##os.system("bash /data/lrq/edge_dection/blurvmaf_sum.bat %s %s" % (name, scale))  #3.blur vmaf; imput video name: Jockey_000

V = read_xls(path, name)
# print("blur patch vmaf is:  \n"+str(V))
L = get_loc(V, C)  # 4.
# print(L)
save_loc(path, name, L)  # 5.

imgH = cv2.imread(path + str(name) + '/sharp/001.png')
imgL = cv2.imread('/home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/' + str(name) + '/LR_bicubic/X' + scale + '/001.png')
print('LR shape:  ' + str(imgL.shape))
if (not os.path.exists(path + str(name) + '/sharp_patch/')):
    os.makedirs(path + str(name) + '/sharp_patch/')
if (not os.path.exists(path + str(name) + '/LR_patch/')):
    os.makedirs(path + str(name) + '/LR_patch/')
for i in range(len(L)):  ###########10 patch   i,j
    ix = L[i][0]
    iy = L[i][1]
    get_patch(path, imgH, imgL, i, ix, iy, int(patch_size), int(scale))  # 6.
print("———————————————Finish Best Patch———————————————")
