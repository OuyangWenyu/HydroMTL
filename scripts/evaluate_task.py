"""
Author: Wenyu Ouyang
Date: 2023-04-05 20:57:26
LastEditTime: 2024-04-15 16:02:22
LastEditors: Wenyu Ouyang
Description: Evaluate the trained model
FilePath: \HydroMTL\scripts\evaluate_task.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import glob
import argparse
import os


from scripts.app_constant import ET_MODIS_NAME, Q_CAMELS_US_NAME
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
    gage_id_file = args.gage_id_file
    if gage_id_file is None or gage_id_file == "None":
        gage_id_file = os.path.join(
            "results", "camels_us_mtl_2001_2021_flow_screen.csv"
        )
    predict_new_mtl_exp(
        exp=exp,
        targets=[Q_CAMELS_US_NAME, ET_MODIS_NAME],
        loss_weights=loss_weight,
        weight_path=weight_path,
        # train_period is just used for a dummy value, not used in the prediction
        train_period=["2001-10-01", "2011-10-01"],
        test_period=test_periods,
        gage_id_file=gage_id_file,
        stat_dict_file=stat_dict_file,
    )


if __name__ == "__main__":
    print("Here are your commands:")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        dest="exp",
        help="the ID of the experiment, such as expstlq001",
        type=str,
        default="expmtl0011",
        # default="expstlq0011",
        # default="expstlet0011",
    )
    parser.add_argument(
        "--loss_weight",
        dest="loss_weight",
        help="weight of loss for usgsFlow or/and ET",
        nargs="+",
        type=float,
        # default=[0.5, 0.5],
        # default=[1, 0],
        default=[0, 1],
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="testing period, such as ['2011-10-01', '2016-10-01']",
        nargs="+",
        # default=["2016-10-01", "2021-10-01"],
        default=["2011-10-01", "2021-10-01"],
    )
    parser.add_argument(
        "--weight_path",
        dest="weight_path",
        help="the weight path file for trained model",
        type=str,
        default=os.path.join(
            "results", "camels", "expmtl001", "12_April_202305_24PM_model.pth"
        ),
        # default=os.path.join("results", "camels", "expstlq001", "07_April_202311_52AM_model.pth"),
        # default=os.path.join("results", "camels", "expstlet001", "09_April_202303_02AM_model.pth"),
    )
    parser.add_argument(
        "--gage_id_file",
        dest="gage_id_file",
        help="the file path of gage id",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    print(f"Your command arguments:{str(args)}")
    train_and_test(args)
