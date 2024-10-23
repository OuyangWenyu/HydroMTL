"""
Author: Wenyu Ouyang
Date: 2024-05-14 17:44:31
LastEditTime: 2024-10-23 14:12:03
LastEditors: Wenyu Ouyang
Description: scripts same with assess_reliability.ipynb
FilePath: \HydroMTL\scripts\assess_reliability.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys

from matplotlib import pyplot as plt


# Get the current directory of the project
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from hydromtl.data.source import data_constant
from hydromtl.explain.probe_analysis import show_probe

# random_seed = 1234
# run_exp_lst = [
#     f"camels{os.sep}expstlq0010",
#     f"camels{os.sep}expmtl0030",
#     f"camels{os.sep}expstlet0010",
# ]
# random_seed = 12345
# run_exp_lst = [
#     f"camels{os.sep}expstlq2020",
#     f"camels{os.sep}expmtl2030",
#     f"camels{os.sep}expstlet0030",
# ]
# random_seed = 123
# run_exp_lst = [
#     f"camels{os.sep}expstlq2030",
#     f"camels{os.sep}expmtl3030",
#     f"camels{os.sep}expstlet0020",
# ]
random_seed = 111
run_exp_lst = [
    f"camels{os.sep}expstlq2040",
    f"camels{os.sep}expmtl4030",
    f"camels{os.sep}expstlet0040",
]
# random_seed = 1111
# run_exp_lst = [
#     f"camels{os.sep}expstlq2050",
#     f"camels{os.sep}expmtl5030",
#     f"camels{os.sep}expstlet0050",
# ]
# set font
plt.rcParams["font.family"] = "Times New Roman"
save_dir = os.path.join(definitions.RESULT_DIR, "figures", f"rs{random_seed}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
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
