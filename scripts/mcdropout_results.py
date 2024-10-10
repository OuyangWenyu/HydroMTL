"""
Author: Wenyu Ouyang
Date: 2024-10-10 18:25:22
LastEditTime: 2024-10-10 19:43:06
LastEditors: Wenyu Ouyang
Description: Run all cases, save the results and plot the results
FilePath: \HydroMTL\scripts\mcdropout_results.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path
import sys


project_dir = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(project_dir)
from scripts.mcdropout_eval import run_mcdropout_for_configs


run_mode = True
# each has 5 random seeds: 1234, 12345, 123, 111, 1111
exp_q_lst = [
    "expstlq001",
    "expstlq202",
    "expstlq203",
    "expstlq204",
    "expstlq205",
]
exp_et_lst = [
    "expstlet001",
    "expstlet003",
    "expstlet002",
    "expstlet004",
    "expstlet005",
]
exp_mtl_lst = [
    "expmtl003",
    "expmtl203",
    "expmtl303",
    "expmtl403",
    "expmtl503",
]
if run_mode:
    # run_mcdropout_for_configs(exp_q_lst, 10)
    run_mcdropout_for_configs(exp_et_lst, 10)
    run_mcdropout_for_configs(exp_mtl_lst, 10)
