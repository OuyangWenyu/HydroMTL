"""
Author: Wenyu Ouyang
Date: 2022-07-23 19:45:55
LastEditTime: 2022-12-21 15:28:46
LastEditors: Wenyu Ouyang
Description: Plots for explaining MTL is better than STL
FilePath: \HydroSPB\hydroSPB\app\multi_task\mtl_better_explain.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydroSPB.data.source import data_constant
from hydroSPB.explore.probe_analysis import show_probe

save_dir = os.path.join(
    definitions.ROOT_DIR,
    "hydroSPB",
    "app",
    "multi_task",
    "results",
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
run_exp_lst = [
    f"camels{os.sep}exp410010",
    f"camels{os.sep}exp410130",
    f"camels{os.sep}exp42001",
]
show_probe(
    run_exp_lst=run_exp_lst,
    var=data_constant.evapotranspiration_modis_camels_us,
    # legend_lst=["STL-Q", "MTL", "STL-ET"],
    legend_lst=["径流单变量模型", "多变量学习模型", "蒸散发单变量模型"],
    show_probe_metric="Corr",
    retrian_probe=[False, False, False],
    num_workers=0,
    save_dir=save_dir,
)

show_probe(
    run_exp_lst=run_exp_lst,
    var=data_constant.streamflow_camels_us,
    legend_lst=["径流单变量模型", "多变量学习模型", "蒸散发单变量模型"],
    show_probe_metric="Corr",
    retrian_probe=[False, False, False],
    num_workers=0,
    save_dir=save_dir,
)

show_probe(
    run_exp_lst=run_exp_lst,
    var=data_constant.surface_soil_moisture_smap_camels_us,
    legend_lst=["径流单变量模型", "多变量学习模型", "蒸散发单变量模型"],
    show_probe_metric="Corr",
    retrian_probe=[False, False, False],
    num_workers=0,
    save_dir=save_dir,
)
