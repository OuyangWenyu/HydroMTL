"""
Author: Wenyu Ouyang
Date: 2024-04-16 09:40:47
LastEditTime: 2024-04-17 09:33:16
LastEditors: Wenyu Ouyang
Description: calculate input probes and compare it with real probe (state ~ output)
FilePath: \HydroMTL\scripts\input_probe.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path
import sys

dir_root = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(dir_root)
import definitions
from hydromtl.explain.probe_analysis import show_probe
from hydromtl.data.source import data_constant


run_exp_lst = [
    f"camels{os.sep}expmtl0010",
]
figure_dir = os.path.join(definitions.RESULT_DIR, "figures")
legend_lst = [
    "MTL",
]
show_probe(
    run_exp_lst=run_exp_lst,
    var=data_constant.streamflow_camels_us,
    legend_lst=legend_lst,
    show_probe_metric="Corr",
    retrian_probe=[False],
    num_workers=0,
    save_dir=figure_dir,
    probe_input=[
        "total_precipitation",
        "temperature",
        "specific_humidity",
        "shortwave_radiation",
        "potential_energy",
    ],
    # probe_input="state",
)
