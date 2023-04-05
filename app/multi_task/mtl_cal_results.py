"""
Author: Wenyu Ouyang
Date: 2022-01-07 22:42:34
LastEditTime: 2023-01-03 21:33:21
LastEditors: Wenyu Ouyang
Description: Get results of multioutput models
FilePath: /HydroSPB/hydroSPB/app/multi_task/mtl_cal_results.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydroSPB.app.streamflow_utils import get_lastest_weight_path
from hydroSPB.app.multi_task.mtl_results_utils import (
    predict_new_q_exp,
    predict_new_mtl_exp,
)
from hydroSPB.data.source.data_constant import (
    ET_MODIS_NAME,
    Q_CAMELS_US_NAME,
    SSM_SMAP_NAME,
)
stl_q_valid_exps = ["exp41001"]
stl_q_test_exps = ["exp410010"]
stl_q_15_21_valid_exps = ["exp41007"]
stl_q_15_21_test_exps = ["exp410070"]
mtl_q_et_ssm_valid_exps = ["exp41085", "exp41091", "exp41097", "exp41103"]
mtl_q_et_ssm_test_exps = [f"{tmp}0" for tmp in mtl_q_et_ssm_valid_exps]
mtl_q_ssm_valid_exps = ["exp41037", "exp41043", "exp41049", "exp41055"]
mtl_q_ssm_test_exps = [f"{tmp}0" for tmp in mtl_q_ssm_valid_exps]
mtl_q_et_valid_exps = ["exp41013", "exp41019", "exp41025", "exp41031"]
mtl_q_et_test_exps = [f"{tmp}0" for tmp in mtl_q_et_valid_exps]
mtl_q_et_15_21_valid_exps = ["exp41061", "exp41067", "exp41073", "exp41079"]
mtl_q_et_15_21_test_exps = [f"{tmp}0" for tmp in mtl_q_et_15_21_valid_exps]
# stl_q_valid_exps = ["exp4247"]
# stl_q_test_exps = ["exp4247"]
# stl_q_15_21_valid_exps = ["exp4234"]
# stl_q_15_21_test_exps = ["exp4234"]
# mtl_q_et_ssm_valid_exps = ["exp4262", "exp4263", "exp4264", "exp4265"]
# mtl_q_et_ssm_test_exps = ["exp4266", "exp4267", "exp4268", "exp4269"]
# mtl_q_ssm_valid_exps = ["exp4235", "exp4236", "exp4237", "exp4238"]
# mtl_q_ssm_test_exps = ["exp4241", "exp4242", "exp4243", "exp4244"]
# mtl_q_et_valid_exps = ["exp4248", "exp4249", "exp4250", "exp4251"]
# mtl_q_et_test_exps = ["exp4270", "exp4271", "exp4272", "exp4273"]
# mtl_q_et_15_21_valid_exps = ["exp4274", "exp4275", "exp4276", "exp4277"]
# mtl_q_et_15_21_test_exps = ["exp4278", "exp4279", "exp4280", "exp4281"]
gage_id_file = os.path.join(
    definitions.ROOT_DIR,
    "hydroSPB",
    "example",
    "camels",
    "camels_us_mtl_2001_2021_flow_screen.csv",
)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

loss_3weights = [
    [0.33, 0.33, 0.33],
    [0.6, 0.2, 0.2],
    [0.8, 0.1, 0.1],
    [0.96, 0.04, 0.04],
]
test3 = False
weight_path_dir3 = [
    os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", "camels", exp)
    for exp in mtl_q_et_ssm_valid_exps
]
weight_path3 = [get_lastest_weight_path(dir_) for dir_ in weight_path_dir3]
if test3:
    cache_path = [None] + [
        os.path.join(
            definitions.ROOT_DIR,
            "hydroSPB",
            "example",
            "camels",
            mtl_q_et_ssm_test_exps[0],
        )
    ] * 3
    for i in range(len(weight_path3)):
        predict_new_mtl_exp(
            exp=mtl_q_et_ssm_test_exps[i],
            targets=[Q_CAMELS_US_NAME, ET_MODIS_NAME, SSM_SMAP_NAME],
            loss_weights=loss_3weights[i],
            weight_path=weight_path3[i],
            train_period=["2015-04-01", "2019-10-01"],
            test_period=["2020-10-01", "2021-10-01"],
            cache_path=cache_path[i],
            gage_id_file=gage_id_file,
        )

loss_weights = [[1.0, 0.0], [0.5, 0.5], [0.75, 0.25], [0.88, 0.11], [0.96, 0.04]]
ssm_test = False
weight_path_dir_ssm = [
    os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", "camels", exp)
    for exp in stl_q_15_21_valid_exps + mtl_q_ssm_valid_exps
]
weight_path_ssm = [get_lastest_weight_path(dir_) for dir_ in weight_path_dir_ssm]
if ssm_test:
    mtl_flow_ssm_exps = stl_q_15_21_test_exps + mtl_q_ssm_test_exps
    cache_path = [None, None] + [
        os.path.join(
            definitions.ROOT_DIR,
            "hydroSPB",
            "example",
            "camels",
            mtl_q_ssm_test_exps[0],
        )
    ] * 3
    for i in range(len(weight_path_ssm)):
        if i == 0:
            predict_new_q_exp(
                exp=mtl_flow_ssm_exps[0],
                weight_path=weight_path_ssm[0],
                train_period=["2015-04-01", "2019-10-01"],
                test_period=["2020-10-01", "2021-10-01"],
                cache_path=cache_path[0],
                gage_id_file=gage_id_file,
            )
        else:
            predict_new_mtl_exp(
                exp=mtl_flow_ssm_exps[i],
                targets=[Q_CAMELS_US_NAME, SSM_SMAP_NAME],
                loss_weights=loss_weights[i],
                weight_path=weight_path_ssm[i],
                train_period=["2015-04-01", "2019-10-01"],
                test_period=["2020-10-01", "2021-10-01"],
                cache_path=cache_path[i],
                gage_id_file=gage_id_file,
            )
test = False
weight_path_dir = [
    os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", "camels", exp)
    for exp in stl_q_valid_exps + mtl_q_et_valid_exps
]
weight_path = [get_lastest_weight_path(dir_) for dir_ in weight_path_dir]
if test:
    mtl_flow_et_exps = stl_q_test_exps + mtl_q_et_test_exps
    cache_path = [None, None] + [
        os.path.join(
            definitions.ROOT_DIR, "hydroSPB", "example", "camels", mtl_q_et_test_exps[0]
        )
    ] * 3
    predict_new_q_exp(
        exp=mtl_flow_et_exps[0],
        weight_path=weight_path[0],
        train_period=["2001-10-01", "2011-10-01"],
        test_period=["2016-10-01", "2021-10-01"],
        cache_path=cache_path[0],
        gage_id_file=gage_id_file,
    )
    for i in range(1, len(weight_path)):
        predict_new_mtl_exp(
            exp=mtl_flow_et_exps[i],
            targets=[Q_CAMELS_US_NAME, ET_MODIS_NAME],
            loss_weights=loss_weights[i],
            weight_path=weight_path[i],
            train_period=["2001-10-01", "2011-10-01"],
            test_period=["2016-10-01", "2021-10-01"],
            cache_path=cache_path[i],
            gage_id_file=gage_id_file,
        )

test_et_15_21 = False
weight_et_15_21_path_dir = [
    os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", "camels", exp)
    for exp in mtl_q_et_15_21_valid_exps
]
weight_et_15_21_path = [
    get_lastest_weight_path(dir_) for dir_ in weight_et_15_21_path_dir
]
loss_weights_et_15_21 = [[0.5, 0.5], [0.75, 0.25], [0.88, 0.11], [0.96, 0.04]]
if test_et_15_21:
    cache_path_15_21 = [None] + [
        os.path.join(
            definitions.ROOT_DIR,
            "hydroSPB",
            "example",
            "camels",
            mtl_q_et_15_21_test_exps[0],
        )
    ] * 3
    for i in range(len(weight_et_15_21_path)):
        predict_new_mtl_exp(
            exp=mtl_q_et_15_21_test_exps[i],
            targets=[Q_CAMELS_US_NAME, ET_MODIS_NAME],
            loss_weights=loss_weights_et_15_21[i],
            weight_path=weight_et_15_21_path[i],
            train_period=["2015-04-01", "2019-10-01"],
            test_period=["2020-10-01", "2021-10-01"],
            cache_path=cache_path_15_21[i],
            gage_id_file=gage_id_file,
        )
