import cv2
import numpy as np
from PIL import Image
from PIL import ImageStat
import math
from matplotlib import pyplot as plt
import glob
import csv
import pandas as pd
import os.path as osp
import time
import threading
import asyncio
import sys
import concurrent.futures
sys.path.append('/home/srteam/lrq/siti-tools')
from siti_tools.siti import SiTiCalculator

def SITI(img, pre_img):
    t = time.time()
    si = SiTiCalculator.si(img)
    t17= time.time() - t
    print("SI:    "+str(t17))

    t = time.time()
    ti = SiTiCalculator.ti(img, pre_img)
    t18= time.time() - t
    print("TI:    "+str(t18))
    return si,ti

def hue(img):
    #path = '/home/srteam/lrq/feature_extract/GT/001.png'
    #img = cv2.imread(path)
    #t=time.time()
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#转化为灰度图
    #t0 = time.time() - t

    #t = time.time()
    h1, s, v = hsv_img[:, :, 0].mean(), hsv_img[:, :, 1].mean(), hsv_img[:, :, 2].mean()#hsv颜色空间
    #t1 = time.time()-t

    #t=time.time()
    h2 = (hsv_img[:, :, 0] - 15) % 180
    h2=h2.mean()
    #t2 = time.time() - t

    #t=time.time()
    h3 = (hsv_img[:, :, 0] - 30) % 180
    h3=h3.mean()
    #t3 = time.time() - t

    #t=time.time()
    h4 = (hsv_img[:, :, 0] - 60) % 180
    h4=h4.mean()
    #t4 = time.time() - t

    #t=time.time()
    h5 = (hsv_img[:, :, 0] - 90) % 180
    h5=h5.mean()
    #t5 = time.time() - t

    #t=time.time()
    h6 = (hsv_img[:, :, 0] - 120) % 180
    h6=h6.mean()
    #t6 = time.time() - t

    #t=time.time()
    h7 = (hsv_img[:, :, 0] - 150 ) % 180
    h7=h7.mean()
    #t7 = time.time() - t
    #print(t0,t1,t2,t3,t4,t5,t6,t7)
    return h1,h2,h3,h4,h5,h6,h7,s,v


def edgelength(img):
    #img = cv2.imread(path)
    t=time.time()

    edges = cv2.Canny(img, 100, 400)
    ###imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ###ret, thresh = cv2.threshold(imgray, 127, 255, 0)  # threshold根据设定的值处理图像的灰度值
    # _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #opencv版本问题，返回两个参数

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)  # contour：带有轮廓信息的图像；cv2.RETR_LIST：以列表形式输出轮廓信息，各轮廓之间无等级关系；CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标；

    perimeter = 0
    for i in contours:
        perimeter += cv2.arcLength(i, True)  # 求长度，识别的contours是否闭合，True对应识别的contours闭合

    t8 = time.time() - t
    print("edgelength:    "+str(t8))
    return perimeter
    # plot contours 绘制等高线
    ###x = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    ###cv2.imshow('Contours', img)
    ###cv2.waitKey(0)
    ###cv2.destroyAllWindows()

# get average Calculate "perceived brightness" of pixels, then return average.
def brightness(path):
   #path = '/home/srteam/lrq/feature_extract/GT/001.png'
   im = Image.open(path)
   t=time.time()

   stat = ImageStat.Stat(im) #ImageStat模块用于计算整个图像或者图像的一个区域的统计数据。
   gs = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)) #确定RGB颜色亮度的公式
         for r,g,b in im.getdata())

   #t9 = time.time() - t
   #print("brightness:    "+str(t9))
   return sum(gs)/stat.count[0]

# get number of keypoints 特征点
def kp_count(img):
    #img = cv2.imread(path)

    t=time.time()

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #sift = cv2.xfeatures2d.SIFT_create() opencv版本问题
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)

    t10 = time.time() - t
    print("kp_count:    "+str(t10))
    return len(kp)

# get contrast 获取两种颜色之间的对比度
def contrast(path):
    #path = '/home/srteam/lrq/feature_extract/GT/001.png'
    img = cv2.imread(path)

    #t=time.time()

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #t11 = time.time() - t
    #print("contrast:    "+str(t11))
    return img_grey.std()

path2 = '/home/srteam/lrq/feature_extract/GT/002.png'
path1 = '/home/srteam/lrq/feature_extract/GT/001.png'
img = cv2.imread(path1)
#t=time.time()
pre_img = cv2.imread(path2)
#t12 = time.time() - t
#print("cv read:    "+str(t12)) #7.8ms

#t=time.time()
im = Image.open(path2)
#t12 = time.time() - t
#print("pil read:    "+str(t12)) #1.7ms

#si,ti=SITI(img, pre_img)
'''
kp= kp_count(img_path)
bright= brightness(im)
ct = contrast(img_path)
el = edgelength(img_path)
h11,h22,h33,h44,h55,h66,h77, ss, vv = hue(img_path)
'''

'''
#16ms NO
thread1 = threading.Thread(target=brightness)
thread2 = threading.Thread(target=contrast)
thread3 = threading.Thread(target=hue)
t=time.time()
thread1.start()
thread2.start()
thread3.start()
thread1.join()
thread2.join()
thread3.join()
t13 = time.time() - t
print("thread:    "+str(t13))

#32ms,NO
async def main():
    await asyncio.gather(brightness(), contrast(), hue())
t=time.time()
asyncio.run(main())
t14 = time.time() - t
print("async:    "+str(t14))
'''
image_path = '/home/srteam/lrq/feature_extract/GT'
def load_and_resize(image_filename):
    img = cv2.imread(image_filename)
    return img

image_files = glob.glob(image_path + '/*.png')
print(len(image_files))
t = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:  ## 默认为1
    res = executor.map(load_and_resize, image_files)
print('多核并行加速后运行 time:', time.time() - t, " 秒")
print("-----end-----")
for i in res:
    print(len(i))

t = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:  ## 默认为1
    res = executor.map(contrast, image_files)
print('多核并行加速后brightness time:', time.time() - t, " 秒")
print("-----end-----")
for i in res:
    print(i)
