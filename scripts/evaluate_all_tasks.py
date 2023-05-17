"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2023-05-16 21:10:07
LastEditors: Wenyu Ouyang
Description: Just try some code
FilePath: /HydroMTL/scripts/evaluate_all_tasks.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import subprocess
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from scripts.streamflow_utils import get_lastest_weight_path

# MTL exps with different λ: 0, ∞, 2, 1, 1/3, 1/8, 1/24
all_valid_exps = [
    "expstlq001",
    "expstlet001",
    "expmtl002",
    "expmtl001",
    "expmtl003",
    "expmtl004",
    "expmtl005",
]
all_test_exps = [f"{tmp}0" for tmp in all_valid_exps]
all_loss_weights = [
    [1, 0],
    [0, 1],
    [0.33, 0.66],
    [0.5, 0.5],
    [0.75, 0.25],
    [0.88, 0.11],
    [0.96, 0.04],
]
all_weight_paths = [
    get_lastest_weight_path(os.path.join(definitions.RESULT_DIR, "camels", tmp))
    for tmp in all_valid_exps
]

args_list = []
for i in range(len(all_test_exps)):
    if i == 0:
        cache_path = "None"
    else:
        cache_path = os.path.join(
            definitions.RESULT_DIR,
            "camels",
            all_test_exps[0],
        )
    args = [
        "python",
        os.path.join(
            definitions.ROOT_DIR,
            "scripts",
            "evaluate_task.py",
        ),
        "--exp",
        all_test_exps[i],
        "--loss_weight",
        str(all_loss_weights[i][0]),
        str(all_loss_weights[i][1]),
        "--test_period",
        "2016-10-01",
        "2021-10-01",
        "--cache_path",
        cache_path,
        "--weight_path",
        all_weight_paths[i],
    ]
    print("Running command: ", " ".join(args))
    args_list.append(args)

print("Please check the command and make sure they are correct.")
input_ = input("Continue running script? (Y/N) ")
if input_.upper() == "Y":
    print("Executing above commands...")
    for args in args_list:
        subprocess.run(args)
elif input_.upper() == "N":
    print("Exiting script.")
    exit()
else:
    print("Invalid input. Enter Y or N.")
    input_ = input("Continue running script? (Y/N) ")
    if input_.upper() == "Y":
        print("Executing above commands...")
        for args in args_list:
            subprocess.run(args)
    elif input_.upper() == "N":
        print("Exiting script.")
        exit()
