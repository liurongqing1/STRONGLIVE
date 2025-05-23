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

def hue(path):
    img = cv2.imread(path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#转化为灰度图
    h1, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]#hsv颜色空间
    h2 = (hsv_img[:, :, 0] - 15) % 180
    h3 = (hsv_img[:, :, 0] - 30) % 180
    h4 = (hsv_img[:, :, 0] - 60) % 180
    h5 = (hsv_img[:, :, 0] - 90) % 180
    h6 = (hsv_img[:, :, 0] - 120) % 180
    h7 = (hsv_img[:, :, 0] - 150 ) % 180
    return h1.mean(),h2.mean(),h3.mean(),h4.mean(),h5.mean(),h6.mean(),h7.mean(),s.mean(),v.mean()


def edgelength(path):
    img = cv2.imread(path)
    edges = cv2.Canny(img, 100, 400)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)  # threshold根据设定的值处理图像的灰度值
    # _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #opencv版本问题，返回两个参数

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)  # contour：带有轮廓信息的图像；cv2.RETR_LIST：以列表形式输出轮廓信息，各轮廓之间无等级关系；CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标；

    perimeter = 0
    for i in contours:
        perimeter += cv2.arcLength(i, True)  # 求长度，识别的contours是否闭合，True对应识别的contours闭合
    return perimeter
    # plot contours 绘制等高线
    x = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# get average Calculate "perceived brightness" of pixels, then return average.
def brightness(path):
   im = Image.open(path)
   stat = ImageStat.Stat(im) #ImageStat模块用于计算整个图像或者图像的一个区域的统计数据。
   gs = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)) #确定RGB颜色亮度的公式
         for r,g,b in im.getdata())
   return sum(gs)/stat.count[0]

# get number of keypoints 特征点
def kp_count(path):
    img = cv2.imread(path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #sift = cv2.xfeatures2d.SIFT_create() opencv版本问题
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    return len(kp)

# get contrast 获取两种颜色之间的对比度
def contrast(path):
    img = cv2.imread(path)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_grey.std()

new_df = pd.DataFrame(
    columns=[ 'name', 'keypoint', 'brightness', 'contrast', 'edgeLength', 'Hue1','Hue2','Hue3','Hue4','Hue5','Hue6','Hue7','Saturation', 'Value'])
#video_names = ["Jockey","Ready","Bosp"]#############


root = '/home/srteam/lrq/feature_extract/GT'
dirs = sorted(glob.glob(root + '/*'))

for dir in dirs:#Bigbosp_002_GT
    kp = bright =ct = el = h1 = h2 = h3 = h4 = h5 = h6= h7= s= v=0
    name = osp.basename(dir)
    files = sorted(glob.glob(dir+ '/*'))
    num = len(files)
    #print(num)
    for img_path in files:#001.png
        kp += kp_count(img_path)
        bright += brightness(img_path)
        ct += contrast(img_path)
        el += edgelength(img_path)
        h11,h22,h33,h44,h55,h66,h77, ss, vv = hue(img_path)
        h1 += h11
        h2 += h22
        h3 += h33
        h4 += h44
        h5 += h55
        h6 += h66
        h7 += h77
        s  += ss
        v  += vv
    '''
        new_df = new_df.append(
            {'name': osp.basename(img_path), 'keypoint': kp , 'brightness': bright , 'contrast': ct , 'edgeLength': el ,
            'Hue1': h11 , 'Hue2': h22 , 'Hue3': h33 , 'Hue4': h44 , 'Hue5': h55 , 'Hue6': h66 ,
            'Hue7': h77, 'Saturation': ss , 'Value': vv }, ignore_index=True)
        print(new_df)
    pd.DataFrame(new_df).to_csv('/home/srteam/lrq/feature_extract/feature_test2_' + name + '.csv', index=None)
    '''
    new_df = new_df.append(
       { 'name': name, 'keypoint': kp/num, 'brightness': bright/num, 'contrast': ct/num, 'edgeLength': el/num,
        'Hue1': h1/num,'Hue2': h2/num,'Hue3': h3/num,'Hue4': h4/num,'Hue5': h5/num,'Hue6': h6/num,
         'Hue7': h7/num, 'Saturation': s/num, 'Value': v/num}, ignore_index=True)
    ##pd.DataFrame(new_df).to_csv('/home/srteam/lrq/feature_extract/feature_test_'+name+'.csv', index=None)
##pd.DataFrame(new_df).to_csv('/home/srteam/lrq/feature_extract/feature_test.csv', index=None)