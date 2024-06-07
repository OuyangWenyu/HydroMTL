"""
Author: Wenyu Ouyang
Date: 2024-05-10 13:26:50
LastEditTime: 2024-06-06 15:01:29
LastEditors: Wenyu Ouyang
Description: This script is used to evaluate the trained models on the training data. We see these results to analyze what the model has learned from the rainfall-runoff data
FilePath: \HydroMTL\scripts\evaluate_train.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import argparse
import os
import sys

# Get the current directory
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from scripts.evaluate_task import train_and_test
from scripts.general_vars import (
    all_train_exps,
    all_valid_exps,
    all_weight_files,
    random_seeds,
)


def evaluate_train_period(
    train_exps=[
        "expstlqtrain001",
        "expstlettrain001",
        "expmtltrain002",
        "expmtltrain001",
        "expmtltrain003",
        "expmtltrain004",
        "expmtltrain005",
    ],
    evaluate_train=True,
    valid_exps=["expstlq001"]
    + ["expstlet001"]
    + ["expmtl002", "expmtl001", "expmtl003", "expmtl004", "expmtl005"],
    weights_file=[
        "07_April_202311_52AM_model.pth",
        "09_April_202303_02AM_model.pth",
        "09_April_202303_57PM_model.pth",
        "12_April_202305_24PM_model.pth",
        "12_April_202306_35PM_model.pth",
        "14_April_202302_40PM_model.pth",
        "14_April_202304_16PM_model.pth",
    ],
    random_seed=1234,
):
    if evaluate_train:
        all_weight_paths = [
            os.path.join(definitions.RESULT_DIR, "camels", exp, weight_file)
            for exp, weight_file in zip(valid_exps, weights_file)
        ]
        loss_weights = [
            [1, 0],
            [0, 1],
            [0.33, 0.66],
            [0.5, 0.5],
            [0.75, 0.25],
            [0.88, 0.11],
            [0.96, 0.04],
        ]
        for exp, weight_path, loss_weight in zip(
            train_exps, all_weight_paths, loss_weights
        ):
            if exp == "expstlqtrain001":
                cache_path = None
                continue
            else:
                cache_path = os.path.join(
                    definitions.RESULT_DIR,
                    "camels",
                    # all the cache path could use the first experiment as only random seed is different
                    "expstlqtrain001",
                )
            args = argparse.Namespace(
                exp=exp,
                loss_weight=loss_weight,
                test_period=["2001-10-01", "2011-10-01"],
                cache_path=cache_path,
                weight_path=weight_path,
                gage_id_file=None,
                n_hidden_states=256,
                layer_hidden_size=128,
                random_seed=random_seed,
                et_product="MOD16A2V006",
            )
            train_and_test(args)


evaluate_trains = [False, True, True, True, True]
for i in range(len(all_train_exps)):
    train_exps = all_train_exps[i]
    evaluate_train = evaluate_trains[i]
    valid_exps = all_valid_exps[i]
    weight_files = all_weight_files[i]
    random_seed = random_seeds[i]
    evaluate_train_period(
        train_exps=train_exps,
        evaluate_train=evaluate_train,
        valid_exps=valid_exps,
        weights_file=weight_files,
        random_seed=random_seed,
    )
