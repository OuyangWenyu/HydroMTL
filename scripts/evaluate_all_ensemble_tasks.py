"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2024-05-26 08:53:30
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
# random_seeds = [1234, 12345, 123, 111, 1111]
random_seeds = [12345, 123, 111, 1111]
all_valid_exps = [
    # random seed 1234
    # [
    #     "expstlq001",
    #     "expstlet001",
    #     "expmtl002",
    #     "expmtl001",
    #     "expmtl003",
    #     "expmtl004",
    #     "expmtl005",
    # ],
    # random seed 12345
    [
        "expstlq202",
        "expstlet003",
        "expmtl202",
        "expmtl201",
        "expmtl203",
        "expmtl204",
        "expmtl205",
    ],
    # random seed 123
    [
        "expstlq203",
        "expstlet002",
        "expmtl302",
        "expmtl301",
        "expmtl303",
        "expmtl304",
        "expmtl305",
    ],
    # random seed 111
    [
        "expstlq204",
        "expstlet004",
        "expmtl402",
        "expmtl401",
        "expmtl403",
        "expmtl404",
        "expmtl405",
    ],
    # random seed 1111
    [
        "expstlq205",
        "expstlet005",
        "expmtl502",
        "expmtl501",
        "expmtl503",
        "expmtl504",
        "expmtl505",
    ],
]
all_weight_files = [
    # random seed 1234
    # [
    #     "07_April_202311_52AM_model.pth",
    #     "09_April_202303_02AM_model.pth",
    #     "09_April_202303_57PM_model.pth",
    #     "12_April_202305_24PM_model.pth",
    #     "12_April_202306_35PM_model.pth",
    #     "14_April_202302_40PM_model.pth",
    #     "14_April_202304_16PM_model.pth",
    # ],
    # random seed 12345
    [
        "21_May_202403_13PM_model.pth",
        "24_April_202309_24PM_model.pth",
        "19_May_202401_01AM_model.pth",
        "18_May_202411_27PM_model.pth",
        "17_May_202404_59PM_model.pth",
        "18_May_202411_23PM_model.pth",
        "19_May_202401_06AM_model.pth",
    ],
    # random seed 123
    [
        "21_May_202408_46PM_model.pth",
        "11_April_202303_45AM_model.pth",
        "19_May_202409_57PM_model.pth",
        "19_May_202409_55PM_model.pth",
        "18_May_202405_18PM_model.pth",
        "19_May_202411_57PM_model.pth",
        "19_May_202411_58PM_model.pth",
    ],
    # random seed 111
    [
        "23_May_202403_18PM_model.pth",
        "24_April_202310_21PM_model.pth",
        "21_May_202409_11PM_model.pth",
        "21_May_202409_11PM_model.pth",
        "21_May_202409_33PM_model.pth",
        "21_May_202411_50PM_model.pth",
        "21_May_202411_51PM_model.pth",
    ],
    # random seed 1111
    [
        "23_May_202403_32PM_model.pth",
        "26_April_202303_04PM_model.pth",
        "23_May_202410_34PM_model.pth",
        "23_May_202409_34PM_model.pth",
        "24_May_202402_13AM_model.pth",
        "24_May_202402_33AM_model.pth",
        "24_May_202402_13AM_model.pth",
    ],
]
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
