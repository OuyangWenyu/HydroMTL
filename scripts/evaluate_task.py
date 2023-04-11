"""
Author: Wenyu Ouyang
Date: 2023-04-05 20:57:26
LastEditTime: 2023-04-10 14:55:12
LastEditors: Wenyu Ouyang
Description: Evaluate the trained model
FilePath: /HydroMTL/scripts/evaluate_task.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import argparse
import os
from pathlib import Path
import sys


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from hydromtl.data.source.data_constant import ET_MODIS_NAME, Q_CAMELS_US_NAME
from mtl_results_utils import predict_new_mtl_exp


def train_and_test(args):
    weight_path = args.weight_path
    if weight_path is None:
        raise ValueError("weight_path is required")
    exp = args.exp
    loss_weight = args.loss_weight
    test_periods = args.test_period
    cache_dir = args.cache_path
    if cache_dir is None or cache_dir=="None":
        cache_dir = os.path.join(definitions.RESULT_DIR, "camels", exp)
    predict_new_mtl_exp(
        exp=exp,
        targets=[Q_CAMELS_US_NAME, ET_MODIS_NAME],
        loss_weights=loss_weight,
        weight_path=weight_path,
        # train_period is just used for a dummy value, not used in the prediction
        train_period=["2001-10-01", "2011-10-01"],
        test_period=test_periods,
        cache_path=cache_dir,
        gage_id_file=os.path.join(
            definitions.RESULT_DIR, "camels_us_mtl_2001_2021_flow_screen.csv"
        ),
    )


if __name__ == "__main__":
    print("Here are your commands:")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        dest="exp",
        help="the ID of the experiment, such as expstlq001",
        type=str,
        default="expmtl0010",
        # default="expstlq0010",
    )
    parser.add_argument(
        "--loss_weight",
        dest="loss_weight",
        help="weight of loss for usgsFlow or/and ET",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
        # default=[1, 0],
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="testing period, such as ['2011-10-01', '2016-10-01']",
        nargs="+",
        default=["2016-10-01", "2021-10-01"],
    )
    parser.add_argument(
        "--cache_path",
        dest="cache_path",
        help="the cache file for forcings, attributes and targets data",
        type=str,
        default=None,
        # default="/mnt/data/owen411/code/HydroMTL/results/camels/expstlq0010",
    )
    parser.add_argument(
        "--weight_path",
        dest="weight_path",
        help="the weight path file for trained model",
        type=str,
        default="/mnt/data/owen411/code/HydroMTL/results/camels/expmtl001/03_August_202201_51AM_model.pth",
        # default="/mnt/data/owen411/code/HydroMTL/results/camels/expstlq001/07_April_202311_52AM_model.pth",
    )
    args = parser.parse_args()
    print(f"Your command arguments:{str(args)}")
    train_and_test(args)
