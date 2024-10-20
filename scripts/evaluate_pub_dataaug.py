"""
Author: Wenyu Ouyang
Date: 2024-10-14 20:34:23
LastEditTime: 2024-10-20 12:07:26
LastEditors: Wenyu Ouyang
Description: evaluate the results
FilePath: \HydroMTL\scripts\evaluate_pub_dataaug.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

# Get the current directory
import os
import sys

from matplotlib import pyplot as plt
import numpy as np


project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from hydromtl.utils import hydro_constant
from hydromtl.models.trainer import stat_result
from scripts.streamflow_utils import get_json_file, plot_ecdf_func
from hydromtl.visual.plot_stat import plot_scatter_with_11line

# we have 2-fold results to see, one-fold for some basins and the other for the rest
# so we directly compare all basins results
exps_lst = [["exppubstlq8010", "exppubstlq8020"], ["exppubmtl7010", "exppubmtl7020"]]
inds_all_lst = []
for item in exps_lst:
    inds_lst = []
    for exp in item:
        cfg_dir_flow_other = os.path.join(definitions.RESULT_DIR, "camels", exp)
        cfg_flow_other = get_json_file(cfg_dir_flow_other)
        inds_df, pred, obs = stat_result(
            cfg_flow_other["data_params"]["test_path"],
            cfg_flow_other["evaluate_params"]["test_epoch"],
            fill_nan=cfg_flow_other["evaluate_params"]["fill_nan"],
            var_unit=[hydro_constant.streamflow.unit],
            var_name=[hydro_constant.streamflow.name],
        )
        inds_lst.append(inds_df[0]["NSE"].values)
    inds_lst_concat = np.array([item for sublist in inds_lst for item in sublist])
    inds_all_lst.append(inds_lst_concat)
plot_ecdf_func(
    inds_all_lst,
    ["STL_Q_PUB", "MTL_PUB"],
    save_path=os.path.join(cfg_dir_flow_other, "pub_nse_comp.png"),
)
plot_scatter_with_11line(
    inds_all_lst[0],
    inds_all_lst[1],
    # xlabel="NSE single-task",
    # ylabel="NSE multi-task",
    xlabel="STL_Q_PUB NSE",
    ylabel="MTL_PUB NSE",
)
plt.savefig(
    os.path.join(
        cfg_dir_flow_other,
        "mtl_stl_pub_q_scatter_plot_with_11line.png",
    ),
    dpi=600,
    bbox_inches="tight",
)
