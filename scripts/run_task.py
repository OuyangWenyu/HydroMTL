"""
Author: Wenyu Ouyang
Date: 2022-04-27 10:54:32
LastEditTime: 2023-04-08 10:26:55
LastEditors: Wenyu Ouyang
Description: Generate commands to run scripts in Linux Screen
FilePath: /HydroMTL/scripts/run_task.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import argparse
import os
from pathlib import Path
import sys


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from mtl_results_utils import run_mtl_camels_flow_et


def train_and_test(args):
    exp = args.exp
    loss_weight = args.loss_weight
    train_periods = args.train_period
    test_periods = args.test_period
    limit_parts = args.limit_part
    ctxs = args.ctx
    random_seed = args.random
    cache_dir = args.cache_path
    if cache_dir is None:
        cache_dir = os.path.join(definitions.RESULT_DIR, "camels", exp)
    run_mtl_camels_flow_et(
        exp,
        random_seed=random_seed,
        cache_dir=cache_dir,
        ctx=ctxs,
        weight_ratio=loss_weight,
        limit_part=limit_parts,
        train_period=train_periods,
        test_period=test_periods,
    )


if __name__ == "__main__":
    print("Here are your commands:")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        dest="exp",
        help="the ID of the experiment, such as expstlq001",
        type=str,
        default="expmtl001",
        # default="expstlet001",
    )
    parser.add_argument(
        "--loss_weight",
        dest="loss_weight",
        help="weight of loss for usgsFlow or/and ET",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
        # default=[0, 1],
    )
    parser.add_argument(
        "--train_period",
        dest="train_period",
        help="training period, such as ['2001-10-01', '2011-10-01']",
        nargs="+",
        default=["2001-10-01", "2011-10-01"],
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="testing period, such as ['2011-10-01', '2016-10-01']",
        nargs="+",
        default=["2011-10-01", "2016-10-01"],
    )
    parser.add_argument(
        "--ctx", dest="ctx", help="CUDA IDs", nargs="+", type=int, default=[0]
    )
    parser.add_argument(
        "--random", dest="random", help="random seed", type=int, default=1234
    )
    parser.add_argument(
        "--limit_part",
        dest="limit_part",
        help="if an output variable is ignored, then its index in all outputs are used.\
            For example,two variables' index are 0 and 1. If first is ignored, then the limit value is 0",
        nargs="+",
        type=int,
        default=None,
        # default=[0],
    )
    parser.add_argument(
        "--cache_path",
        dest="cache_path",
        help="the cache file for forcings, attributes and targets data",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    print(f"Your command arguments:{str(args)}")
    train_and_test(args)
