"""
This script is used to evaluate the trained models on the training data.
We see these results to analyze what the model has learned from the rainfall-runoff data
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

mtl_q_et_exps = ["expmtl002", "expmtl001", "expmtl003", "expmtl004", "expmtl005"]
all_exps = ["expstlq001"] + ["expstlet001"] + mtl_q_et_exps
train_exps = [
    "expstlqtrain001",
    "expstlettrain001",
    "expmtltrain002",
    "expmtltrain001",
    "expmtltrain003",
    "expmtltrain004",
    "expmtltrain005",
]
evaluate_train = True
if evaluate_train:
    all_weight_paths = [
        os.path.join(
            definitions.RESULT_DIR,
            "camels",
            "expstlq001",
            "07_April_202311_52AM_model.pth",
        ),
        os.path.join(
            definitions.RESULT_DIR,
            "camels",
            "expstlet001",
            "09_April_202303_02AM_model.pth",
        ),
        os.path.join(
            definitions.RESULT_DIR,
            "camels",
            "expmtl002",
            "09_April_202303_57PM_model.pth",
        ),
        os.path.join(
            definitions.RESULT_DIR,
            "camels",
            "expmtl001",
            "12_April_202305_24PM_model.pth",
        ),
        os.path.join(
            definitions.RESULT_DIR,
            "camels",
            "expmtl003",
            "12_April_202306_35PM_model.pth",
        ),
        os.path.join(
            definitions.RESULT_DIR,
            "camels",
            "expmtl004",
            "14_April_202302_40PM_model.pth",
        ),
        os.path.join(
            definitions.RESULT_DIR,
            "camels",
            "expmtl005",
            "14_April_202304_16PM_model.pth",
        ),
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
                "expstlqtrain001",
            )
        args = argparse.Namespace(
            exp=exp,
            loss_weight=loss_weight,
            test_period=["2001-10-01", "2011-10-01"],
            cache_path=cache_path,
            weight_path=weight_path,
            gage_id_file=None,
        )
        train_and_test(args)
