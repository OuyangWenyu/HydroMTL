"""
Author: Wenyu Ouyang
Date: 2023-04-05 20:57:26
LastEditTime: 2024-05-29 15:26:45
LastEditors: Wenyu Ouyang
Description: Evaluate the trained model
FilePath: \HydroMTL\scripts\evaluate_task.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import glob
import argparse
import os
from pathlib import Path
import sys

project_dir = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(project_dir)
import definitions
from hydromtl.data.source.data_constant import ET_MODIS_NAME, Q_CAMELS_US_NAME
from scripts.mtl_results_utils import predict_new_mtl_exp


def train_and_test(args):
    weight_path = args.weight_path
    if weight_path is None:
        raise ValueError("weight_path is required")
    train_exp_dir = os.sep.join(weight_path.split(os.sep)[:-1])
    stat_dict_file = glob.glob(os.path.join(train_exp_dir, "*_stat.json"))[0]
    exp = args.exp
    loss_weight = args.loss_weight
    test_periods = args.test_period
    cache_dir = args.cache_path
    if cache_dir is None or cache_dir == "None":
        cache_dir = os.path.join(definitions.RESULT_DIR, "camels", exp)
    gage_id_file = args.gage_id_file
    if gage_id_file is None or gage_id_file == "None":
        gage_id_file = os.path.join(
            definitions.RESULT_DIR, "camels_us_mtl_2001_2021_flow_screen.csv"
        )
    n_hidden_states = args.n_hidden_states
    layer_hidden_size = args.layer_hidden_size
    random_seed = args.random_seed
    et_product = args.et_product
    predict_new_mtl_exp(
        exp=exp,
        targets=[Q_CAMELS_US_NAME, ET_MODIS_NAME],
        loss_weights=loss_weight,
        weight_path=weight_path,
        # train_period is just used for a dummy value, not used in the prediction
        train_period=["2001-10-01", "2011-10-01"],
        test_period=test_periods,
        cache_path=cache_dir,
        gage_id_file=gage_id_file,
        stat_dict_file=stat_dict_file,
        n_hidden_states=n_hidden_states,
        layer_hidden_size=layer_hidden_size,
        random_seed=random_seed,
        et_product=et_product,
    )


if __name__ == "__main__":
    print("Here are your commands:")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        dest="exp",
        help="the ID of the experiment, such as expstlq001",
        type=str,
        default="expmtl2030",
        # default="expstlq2030",
        # default="expstlet9010",
        # default="expstlet0030",
    )
    parser.add_argument(
        "--loss_weight",
        dest="loss_weight",
        help="weight of loss for usgsFlow or/and ET",
        nargs="+",
        type=float,
        # default=[0.5, 0.5],
        default=[0.75, 0.25],
        # default=[1, 0],
        # default=[0, 1],
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="testing period, such as ['2011-10-01', '2016-10-01']",
        nargs="+",
        default=["2016-10-01", "2021-10-01"],
        # default=["2011-10-01", "2021-10-01"],
    )
    parser.add_argument(
        "--cache_path",
        dest="cache_path",
        help="the cache file for forcings, attributes and targets data",
        type=str,
        # default=None,
        default=os.path.join(
            definitions.RESULT_DIR,
            "camels",
            "expstlq0010",  # no matter mtl or stl, cache is for both vars
        ),
    )
    parser.add_argument(
        "--weight_path",
        dest="weight_path",
        help="the weight path file for trained model",
        type=str,
        default=os.path.join(
            definitions.RESULT_DIR,
            "camels",
            # "expstlet901",
            # "28_May_202409_28PM_model.pth",
            "expstlq203",
            "18_May_202410_44AM_model.pth",
        ),
    )
    parser.add_argument(
        "--gage_id_file",
        dest="gage_id_file",
        help="the file path of gage id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n_hidden_states",
        dest="n_hidden_states",
        help="the number of hidden states for LSTM model",
        type=int,
        default=256,
        # default=64,
    )
    parser.add_argument(
        "--layer_hidden_size",
        dest="layer_hidden_size",
        help="the number of hidden size for output layer",
        type=int,
        default=128,
        # default=32,
    )
    parser.add_argument(
        "--random_seed",
        dest="random_seed",
        help="the random seed for the model",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--et_product",
        dest="et_product",
        help="the product of ET",
        type=str,
        default="MOD16A2V006",
        # default="MOD16A2GFV061",
    )
    args = parser.parse_args()
    print(f"Your command arguments:{str(args)}")
    train_and_test(args)
