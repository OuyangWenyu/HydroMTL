"""
Author: Wenyu Ouyang
Date: 2024-05-14 17:44:31
LastEditTime: 2024-05-14 17:44:46
LastEditors: Wenyu Ouyang
Description: scripts same with assess_reliability.ipynb
FilePath: \HydroMTL\scripts\assess_reliability.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys


# Get the current directory of the project
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from hydromtl.data.source import data_constant
from hydromtl.explain.probe_analysis import show_probe

save_dir = os.path.join(
    definitions.RESULT_DIR,
    "figures",
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
run_exp_lst = [
    f"camels{os.sep}expstlq0010",
    f"camels{os.sep}expmtl0030",
    f"camels{os.sep}expstlet0010",
]
legend_lst = ["STL-Q", "MTL", "STL-ET"]
# First probe is for evapotranspiration (ET).
show_probe(
    run_exp_lst=run_exp_lst,
    var=data_constant.evapotranspiration_modis_camels_us,
    legend_lst=legend_lst,
    show_probe_metric="Corr",
    retrian_probe=[False, False, False],
    num_workers=0,
    save_dir=save_dir,
)
# The second probe is for streamflow (Q).
show_probe(
    run_exp_lst=run_exp_lst,
    var=data_constant.streamflow_camels_us,
    legend_lst=legend_lst,
    show_probe_metric="Corr",
    retrian_probe=[False, False, False],
    num_workers=0,
    save_dir=save_dir,
)
# The final probe is for soil moisture (SM).
show_probe(
    run_exp_lst=run_exp_lst,
    var=data_constant.surface_soil_moisture_smap_camels_us,
    legend_lst=legend_lst,
    show_probe_metric="Corr",
    retrian_probe=[False, False, False],
    num_workers=0,
    save_dir=save_dir,
)