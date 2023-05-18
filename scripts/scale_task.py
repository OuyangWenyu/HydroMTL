"""
Author: Wenyu Ouyang
Date: 2023-05-16 20:48:04
LastEditTime: 2023-05-18 17:39:46
LastEditors: Wenyu Ouyang
Description: See scaling effect of MTL and STL exps
FilePath: /HydroMTL/scripts/scale_task.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import argparse
import glob
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from streamflow_utils import get_lastest_weight_path
from mtl_results_utils import predict_new_mtl_exp, run_mtl_camels
from hydromtl.data.source.data_constant import ET_MODIS_NAME, Q_CAMELS_US_NAME


# set some tasks to evaluate the scaling behavior of MTL and STL exps
# each task train models on different numbers of basins: 1%, 5%, 10%, 25%, 30%, 50%, 75%, 100%
# randomly select x% of basins from the training set
def random_select_basins(x_percent, n_basins, random_seed=1234):
    # Calculate number to choose
    n_choose = int(n_basins * x_percent)
    np.random.seed(random_seed)
    inds = np.random.choice(n_basins, size=n_choose, replace=False)
    return np.sort(inds)


def scaling_exp(args):
    exp = args.exp
    targets = args.output_vars
    loss_weight = args.loss_weight
    train_periods = args.train_period
    test_periods = args.test_period
    limit_parts = args.limit_part
    ctxs = args.ctx
    random_seed = args.random_seed
    x_percents = args.x_percent
    scaling_exps = [exp + str(x_percents[i]).zfill(3) for i in range(len(x_percents))]
    all_basins = pd.read_csv(
        os.path.join(definitions.RESULT_DIR, "camels_us_mtl_2001_2021_flow_screen.csv"),
        dtype={"GAGE_ID": str},
    )
    for i in range(len(scaling_exps)):
        chosen_idx = random_select_basins(
            x_percents[i] / 100, all_basins.shape[0], random_seed
        )
        gage_ids = all_basins.loc[chosen_idx].values[:, 0].tolist()
        run_mtl_camels(
            scaling_exps[i],
            targets=targets,
            weight_ratio=loss_weight,
            train_period=train_periods,
            test_period=test_periods,
            gage_id=gage_ids,
            gage_id_file=None,
            limit_part=limit_parts,
            ctx=ctxs,
        )


if __name__ == "__main__":
    print("Here are your commands:")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        dest="exp",
        help="the ID of the experiment, such as exppub00",
        type=str,
        default="expscalemtl",
    )
    parser.add_argument(
        "--x_percent",
        dest="x_percent",
        help="randomly select x percent of basins from the training set",
        nargs="+",
        type=int,
        default=[1, 5, 10, 25, 30, 50, 75, 100],
    )
    parser.add_argument(
        "--random_seed",
        dest="random_seed",
        help="random seed for randomly selecting basins",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--output_vars",
        dest="output_vars",
        help="the variables as output of MTL models, chosen from ['usgsFlow', 'ET']",
        nargs="+",
        type=str,
        default=[Q_CAMELS_US_NAME, ET_MODIS_NAME],
    )
    parser.add_argument(
        "--loss_weight",
        dest="loss_weight",
        help="weight of loss for usgsFlow or/and ET/ssm",
        nargs="+",
        type=float,
        default=[0.75, 0.25],
        # default=[0.0, 1.0],
        # default=[1.0, 0.0],
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
        help="testing period, such as ['2011-10-01', '2021-10-01']",
        nargs="+",
        default=["2011-10-01", "2021-10-01"],
    )
    parser.add_argument(
        "--ctx", dest="ctx", help="CUDA IDs", nargs="+", type=int, default=[0]
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
        # default=[1],
    )
    args = parser.parse_args()
    print(f"Your command arguments:{str(args)}")
    scaling_exp(args)
