"""
Author: Wenyu Ouyang
Date: 2024-10-10 21:01:43
LastEditTime: 2024-10-22 08:52:04
LastEditors: Wenyu Ouyang
Description: evaluate all pub cases and plot the results
FilePath: \HydroMTL\scripts\evaluate_improve4lessobspub.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

# Get the current directory of the project
import glob
import json
import os
import sys


project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from hydromtl.data.source.data_constant import (
    ET_MODIS_NAME,
    Q_CAMELS_US_NAME,
    SSM_SMAP_NAME,
)
from scripts.mtl_results_utils import predict_new_mtl_exp
from scripts.streamflow_utils import get_json_file, get_lastest_weight_path


def evaluate_pub(train_exp, gage_id_file, cache_dir=None, second_var=SSM_SMAP_NAME):
    train_exp_dir = os.path.join(definitions.RESULT_DIR, "camels", train_exp)
    weight_path = get_lastest_weight_path(train_exp_dir)
    if weight_path is None:
        raise ValueError("weight_path is required")
    stat_dict_file = glob.glob(os.path.join(train_exp_dir, "*_stat.json"))[0]
    config = get_json_file(train_exp_dir)
    new_exp = f"{train_exp}0"
    predict_new_mtl_exp(
        exp=new_exp,
        targets=[Q_CAMELS_US_NAME, second_var],
        loss_weights=config["training_params"]["criterion_params"]["item_weight"],
        weight_path=weight_path,
        train_period=config["data_params"]["t_range_train"],
        test_period=config["data_params"]["t_range_test"],
        cache_path=cache_dir,
        gage_id_file=gage_id_file,
        stat_dict_file=stat_dict_file,
        n_hidden_states=config["model_params"]["model_param"]["n_hidden_states"],
        layer_hidden_size=config["model_params"]["model_param"]["layer_hidden_size"],
        random_seed=config["training_params"]["random_seed"],
        et_product="MOD16A2V006",
    )


run_mode = False
train_exppubstlq_fold1 = [
    "exppubstlq801",
    "exppubstlq803",
    "exppubstlq805",
    "exppubstlq807",
    "exppubstlq809",
]
train_exppubmtl_fold1 = [
    "exppubmtl701",
    "exppubmtl703",
    "exppubmtl705",
    "exppubmtl707",
    "exppubmtl709",
]
train_exppubstlq_fold2 = [
    "exppubstlq802",
    "exppubstlq804",
    "exppubstlq806",
    "exppubstlq808",
    "exppubstlq810",
]
train_exppubmtl_fold2 = [
    "exppubmtl702",
    "exppubmtl704",
    "exppubmtl706",
    "exppubmtl708",
    "exppubmtl710",
]
if run_mode:
    fold1_test_gage_id_file = os.path.join(
        definitions.RESULT_DIR, "exp_pub_kfold_percent050", "camels_test_kfold0.csv"
    )
    for train_exp in train_exppubstlq_fold1:
        evaluate_pub(
            train_exp=train_exp,
            gage_id_file=fold1_test_gage_id_file,
            # cache is for the pub-test basins which are used for traineing in another fold experiment
            cache_dir=os.path.join(definitions.RESULT_DIR, "camels", "exppubstlq802"),
        )
    for train_exp in train_exppubmtl_fold1:
        evaluate_pub(
            train_exp=train_exp,
            gage_id_file=fold1_test_gage_id_file,
            cache_dir=os.path.join(definitions.RESULT_DIR, "camels", "exppubstlq802"),
        )
    fold2_test_gage_id_file = os.path.join(
        definitions.RESULT_DIR, "exp_pub_kfold_percent050", "camels_test_kfold1.csv"
    )
    for train_exp in train_exppubstlq_fold2:
        evaluate_pub(
            train_exp=train_exp,
            gage_id_file=fold2_test_gage_id_file,
            cache_dir=os.path.join(definitions.RESULT_DIR, "camels", "exppubstlq801"),
        )
    for train_exp in train_exppubmtl_fold2:
        evaluate_pub(
            train_exp=train_exp,
            gage_id_file=fold2_test_gage_id_file,
            cache_dir=os.path.join(definitions.RESULT_DIR, "camels", "exppubstlq801"),
        )
