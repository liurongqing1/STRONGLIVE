import numpy as np
import glob
import csv
import pandas as pd
import os.path as osp
import json

txt_tables = []
f = open('/home/srteam/lrq/feature_extract/GT_path.out', "r",encoding='utf-8')
line = f.readline() # 读取第一行
while line:
    txt_data = line[:-1]# 可将字符串变为元组
    txt_tables.append(txt_data) # 列表增加
    line = f.readline() # 读取下一行
print(len(txt_tables))
#print(txt_tables)
txt_tables = json.dumps(txt_tables)#变双引号
print(txt_tables)

