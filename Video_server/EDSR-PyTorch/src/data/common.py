import random

import numpy as np
import skimage.color as sc

import torch

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):#########随机补丁
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size#输出大小
        ip = tp // scale#输入大小
    else:
        tp = patch_size#lr,hr同大小
        ip = patch_size


    #原
    ix = random.randrange(0, iw - ip + 1)#返回给定范围内的随机整数
    iy = random.randrange(0, ih - ip + 1)

    '''
    ix = 200#random.randrange(0, iw - ip + 1)#返回给定范围内的随机整数
    iy = 60#20#random.randrange(0, ih - ip + 1)
    '''
    if not input_large:
        tx, ty = scale * ix, scale * iy#等比例映射
    else:
        tx, ty = ix, iy
    #原
    #lr：[10:10+200,10:10+200], hr：[60:60+1200,60:60+1200]
    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]
    '''
    #灰度图改
    ret = [
        args[0][iy:iy + ip, ix:ix + ip],
        *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]
    ]
    '''
    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    #原
    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img
    '''
    #灰度图改
    def _augment(img):
        if hflip: img = img[:, ::-1 ]
        if vflip: img = img[::-1, :]
        if rot90: img = img.transpose(1, 0, 2)#（channels,imagesize,imagesize）转化为（imagesize,imagesize,channels）

        return img
    '''
    return [_augment(a) for a in args]

