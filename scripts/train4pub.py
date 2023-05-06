"""
Author: Wenyu Ouyang
Date: 2022-12-14 15:05:47
LastEditTime: 2023-05-05 14:52:46
LastEditors: Wenyu Ouyang
Description: Run MTL experiments for PUB experiments
FilePath: /HydroMTL/scripts/train4pub.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""


import argparse
import glob
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from scripts.mtl_results_utils import predict_new_mtl_exp, run_mtl_camels_flow_et
from hydromtl.data.source.data_constant import ET_MODIS_NAME, Q_CAMELS_US_NAME
from scripts.streamflow_utils import get_lastest_weight_path


def pub_train_and_test(args):
    exp = args.exp
    targets = args.output_vars
    loss_weight = args.loss_weight
    train_periods = args.train_period
    test_periods = args.test_period
    limit_parts = args.limit_part
    ctxs = args.ctx
    split_num = 2
    pub_exps = [exp + str(i + 1) for i in range(split_num)]
    for i in range(len(pub_exps)):
        run_mtl_camels_flow_et(
            pub_exps[i],
            targets=targets,
            weight_ratio=loss_weight,
            train_period=train_periods,
            test_period=test_periods,
            gage_id_file=os.path.join(
                definitions.RESULT_DIR,
                f"exp_pub_kfold{split_num}",
                f"camels_train_kfold{str(i)}.csv",
            ),
            limit_part=limit_parts,
            ctx=ctxs,
        )
        stat_dict_file = glob.glob(
            os.path.join(definitions.RESULT_DIR, "camels", pub_exps[i], "*_stat.json")
        )[0]
        weight_path_dir = os.path.join(definitions.RESULT_DIR, "camels", pub_exps[i])
        weight_path = get_lastest_weight_path(weight_path_dir)
        predict_new_mtl_exp(
            exp=f"{pub_exps[i]}0",
            targets=targets,
            loss_weights=loss_weight,
            weight_path=weight_path,
            train_period=train_periods,
            test_period=test_periods,
            gage_id_file=os.path.join(
                definitions.RESULT_DIR,
                f"exp_pub_kfold{split_num}",
                f"camels_test_kfold{str(i)}.csv",
            ),
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
        default="exppub00",
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
        default=[0.5, 0.5],
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
    pub_train_and_test(args)
