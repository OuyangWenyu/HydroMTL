"""
Author: Wenyu Ouyang
Date: 2024-05-15 14:07:41
LastEditTime: 2024-05-15 19:30:03
LastEditors: Wenyu Ouyang
Description: 
FilePath: \HydroMTL\tests\test_hydro_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path
import sys
import time
import numpy as np
import torch


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from hydromtl.utils.hydro_utils import deal_gap_data_old, deal_gap_data


def test_deal_gap_data_1():
    # 创建示例数据
    output = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ],
        device="cpu",
    )
    target = torch.tensor(
        [
            [1.0, float("nan"), float("nan")],
            [float("nan"), 5.0, float("nan")],
            [7.0, float("nan"), 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ],
        device="cpu",
    )
    data_gap = 1
    device = torch.device("cpu")

    # 使用 deal_gap_data_old 函数
    p_old, t_old = deal_gap_data_old(output, target, data_gap, device)
    print("Old function output p:", p_old)
    print("Old function output t:", t_old)

    # 使用 deal_gap_data 函数
    p_new, t_new = deal_gap_data(output, target, data_gap, device)
    print("New function output p:", p_new)
    print("New function output t:", t_new)

    # 比较两个函数的输出
    assert torch.allclose(p_old, p_new), "Output p does not match!"
    assert torch.allclose(t_old, t_new), "Output t does not match!"


def test_deal_gap_data_2():
    output = torch.tensor(
        [
            [-0.0005, -0.0429, -0.0342, -0.0472, -0.0435],
            [0.0091, -0.0539, -0.0474, -0.0611, -0.0530],
            [0.0076, -0.0630, -0.0485, -0.0562, -0.0527],
            [0.0024, -0.0592, -0.0569, -0.0590, -0.0533],
            [0.0054, -0.0659, -0.0575, -0.0567, -0.0506],
            [0.0112, -0.0493, -0.0600, -0.0644, -0.0539],
            [0.0190, -0.0554, -0.0642, -0.0697, -0.0602],
            [0.0159, -0.0426, -0.0631, -0.0695, -0.0582],
            [0.0066, -0.0473, -0.0580, -0.0599, -0.0592],
            [0.0058, -0.0461, -0.0611, -0.0586, -0.0588],
            [0.0152, -0.0466, -0.0628, -0.0651, -0.0605],
            [0.0075, -0.0556, -0.0642, -0.0650, -0.0585],
            [-0.0011, -0.0458, -0.0583, -0.0693, -0.0567],
            [0.0034, -0.0416, -0.0643, -0.0714, -0.0588],
            [0.0035, -0.0564, -0.0668, -0.0585, -0.0586],
            [0.0089, -0.0597, -0.0634, -0.0518, -0.0548],
            [0.0108, -0.0684, -0.0525, -0.0605, -0.0528],
            [0.0171, -0.0580, -0.0635, -0.0620, -0.0591],
            [0.0021, -0.0636, -0.0689, -0.0651, -0.0571],
            [0.0010, -0.0584, -0.0635, -0.0723, -0.0522],
            [0.0011, -0.0695, -0.0542, -0.0556, -0.0570],
            [-0.0033, -0.0794, -0.0551, -0.0407, -0.0565],
            [0.0075, -0.0846, -0.0510, -0.0582, -0.0599],
            [0.0086, -0.0883, -0.0489, -0.0637, -0.0625],
            [-0.0079, -0.0871, -0.0586, -0.0487, -0.0544],
            [-0.0025, -0.0820, -0.0668, -0.0412, -0.0598],
            [0.0052, -0.0613, -0.0621, -0.0448, -0.0654],
            [-0.0003, -0.0506, -0.0694, -0.0518, -0.0631],
            [0.0050, -0.0564, -0.0573, -0.0597, -0.0637],
            [0.0007, -0.0581, -0.0546, -0.0727, -0.0635],
        ],
        device="cuda:0",
    )
    target = torch.tensor(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, -0.4325],
            [np.nan, -0.3361, np.nan, 0.8308, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [-0.6852, np.nan, 1.2694, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, -0.5364],
            [np.nan, -0.1330, np.nan, 1.1941, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [-0.4113, np.nan, 1.3453, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 0.1118, np.nan, 0.9350, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [-0.3751, np.nan, 1.2861, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, -0.1223],
            [np.nan, -0.1122, np.nan, 1.4138, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 1.3529, np.nan, np.nan],
        ],
        device="cuda:0",
    )
    data_gap = 2
    device = torch.device("cuda:0")

    # 使用 deal_gap_data_old 函数
    p_old, t_old = deal_gap_data_old(output, target, data_gap, device)
    print("Old function output p:", p_old)
    print("Old function output t:", t_old)

    # 使用 deal_gap_data 函数
    p_new, t_new = deal_gap_data(output, target, data_gap, device)
    print("New function output p:", p_new)
    print("New function output t:", t_new)

    # 比较两个函数的输出
    assert torch.allclose(p_old, p_new), "Output p does not match!"
    assert torch.allclose(t_old, t_new), "Output t does not match!"


def test_time():
    # 创建大规模数据集
    data0 = torch.tensor([2.0]).repeat(8000, 30, 1)
    data1 = torch.tensor(np.full((8000, 30, 1), np.nan))
    for i in range(30):
        for j in range(0, 8000, 1000):
            data1[j, i, :] = (i + 1) * (j // 1000 + 1)
    targ = torch.cat((data0, data1), dim=-1)
    pred = torch.tensor(np.full((8000, 30, 2), 1.0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targ = targ.to(device)
    pred = pred.to(device)

    # 测量原始方法的执行时间
    start_time = time.time()
    p_orig, t_orig = deal_gap_data_old(
        pred[:, :, 0], targ[:, :, 1], data_gap=1, device=device
    )
    end_time = time.time()
    print(f"Original method time: {end_time - start_time:.4f} seconds")

    # 测量优化方法的执行时间
    start_time = time.time()
    p_opt, t_opt = deal_gap_data(
        pred[:, :, 0], targ[:, :, 1], data_gap=1, device=device
    )
    end_time = time.time()
    print(f"Optimized method time: {end_time - start_time:.4f} seconds")
