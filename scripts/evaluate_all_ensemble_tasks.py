"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2024-06-06 14:07:46
LastEditors: Wenyu Ouyang
Description: Just try some code
FilePath: \HydroMTL\scripts\evaluate_all_ensemble_tasks.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import subprocess
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from scripts.streamflow_utils import get_lastest_weight_path
from scripts.general_vars import (
    all_loss_weights,
    all_valid_exps,
    all_weight_files,
    random_seeds,
)

all_test_exps = []
all_weight_paths = []
for i in range(len(all_valid_exps)):
    all_test_exps_ = [f"{tmp}0" for tmp in all_valid_exps[i]]
    all_test_exps.append(all_test_exps_)
    all_weight_paths_ = [
        os.path.join(
            definitions.RESULT_DIR,
            "camels",
            all_valid_exps[i][j],
            all_weight_files[i][j],
        )
        for j in range(len(all_valid_exps[i]))
    ]
    all_weight_paths.append(all_weight_paths_)

args_list = []
for j in range(len(all_test_exps)):
    all_test_exps_ = all_test_exps[j]
    all_weight_paths_ = all_weight_paths[j]
    random_seed = random_seeds[j]
    for i in range(len(all_test_exps_)):
        if i == 0 and j == 0:
            cache_path = "None"
        else:
            cache_path = os.path.join(
                definitions.RESULT_DIR,
                "camels",
                all_test_exps[0][0],
            )
        args = [
            "python",
            os.path.join(
                definitions.ROOT_DIR,
                "scripts",
                "evaluate_task.py",
            ),
            "--exp",
            all_test_exps_[i],
            "--loss_weight",
            str(all_loss_weights[i][0]),
            str(all_loss_weights[i][1]),
            "--test_period",
            "2016-10-01",
            "2021-10-01",
            "--cache_path",
            cache_path,
            "--weight_path",
            all_weight_paths_[i],
            "--random_seed",
            str(random_seed),
        ]
        print("Running command: ", " ".join(args))
        if not os.path.isfile(all_weight_paths_[i]):
            raise FileNotFoundError(f"Weight file {all_weight_paths_[i]} not found.")
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
