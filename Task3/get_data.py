import copy
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件并转换为 DataFrame
def read_csv(file_name):
    return pd.read_csv(file_name)

def build_box_and_num_list(df):
    # 按订单号 (sta_code) 分组
    box_dict = {}
    
    # 遍历每个订单
    for sta_code, group in df.groupby('sta_code'):
        box_list = []
        num_list = []
        total_volume = 0  # 用于存储该订单下所有 Box 的总体积
        
        # 获取当前 sta_code 下的所有唯一 sku_code
        sku_code_mapping = {sku_code: idx for idx, sku_code in enumerate(group['sku_code'].unique())}
        
        box_type = 0
        # 遍历每个物品
        for _, row in group.iterrows():
            lx = row['长(CM)']
            ly = row['宽(CM)']
            lz = row['高(CM)']
            qty = row['qty']
            
            # 创建 Box 对象
            box = Box(lx, ly, lz, box_type)
            box_type += 1
            
            # 计算每个 Box 的体积并累加
            box_volume = lx * ly * lz
            total_volume += box_volume * qty  # 每种物品的体积乘以数量
            
            # 添加到 box_list 和 num_list
            box_list.append(box)
            num_list.append(qty)
        
        # 存储每个订单的 box_list 和 num_list，以及该订单的总物品体积
        box_dict[sta_code] = (box_list, num_list, total_volume)
    
    return box_dict