"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2024-05-25 10:31:24
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

# MTL exps with different λ: 0, ∞, 2, 1, 1/3, 1/8, 1/24
random_seeds = [1234, 12345, 123, 111, 1111]
all_valid_exps = [
    # random seed 1234
    [
        "expstlq001",
        "expstlet001",
        "expmtl002",
        "expmtl001",
        "expmtl003",
        "expmtl004",
        "expmtl005",
    ],
    # random seed 12345
    [
        "expstlq202",
        "expstlet003",
        "expmtl201",
        "expmtl202",
        "expmtl203",
        "expmtl204",
        "expmtl205",
    ],
    # random seed 123
    [
        "expstlq203",
        "expstlet002",
        "expmtl301",
        "expmtl302",
        "expmtl303",
        "expmtl304",
        "expmtl305",
    ],
    # random seed 111
    [
        "expstlq204",
        "expstlet004",
        "expmtl401",
        "expmtl402",
        "expmtl403",
        "expmtl404",
        "expmtl405",
    ],
    # random seed 1111
    [
        "expstlq205",
        "expstlet005",
        "expmtl501",
        "expmtl502",
        "expmtl503",
        "expmtl504",
        "expmtl505",
    ],
]
all_test_exps = []
all_weight_paths = []
for all_valid_exps_ in all_valid_exps:
    all_test_exps_ = [f"{tmp}0" for tmp in all_valid_exps_]
    all_test_exps.append(all_test_exps_)
    all_weight_paths_ = [
        get_lastest_weight_path(os.path.join(definitions.RESULT_DIR, "camels", tmp))
        for tmp in all_valid_exps_
    ]
    all_weight_paths.append(all_weight_paths_)

all_loss_weights = [
    [1, 0],
    [0, 1],
    [0.33, 0.66],
    [0.5, 0.5],
    [0.75, 0.25],
    [0.88, 0.11],
    [0.96, 0.04],
]
args_list = []
for j in range(len(all_test_exps)):
    all_test_exps_ = all_test_exps[j]
    all_weight_paths_ = all_weight_paths[j]
    random_seed = random_seeds[j]
    for i in range(len(all_test_exps_)):
        if i == 0:
            cache_path = "None"
        else:
            cache_path = os.path.join(
                definitions.RESULT_DIR,
                "camels",
                all_test_exps_[0],
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
