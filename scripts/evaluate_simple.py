"""
Author: Wenyu Ouyang
Date: 2024-10-14 20:34:23
LastEditTime: 2024-10-14 21:15:36
LastEditors: Wenyu Ouyang
Description: evaluate the results
FilePath: \HydroMTL\scripts\evaluate_simple.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

# Get the current directory
import os
import sys


project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from hydromtl.utils import hydro_constant
from hydromtl.models.trainer import stat_result
from scripts.streamflow_utils import get_json_file, plot_ecdf_func

exps_lst = ["exppubstlq4010", "exppubmtl5010"]
inds_all_lst = []
for item in exps_lst:
    cfg_dir_flow_other = os.path.join(definitions.RESULT_DIR, "camels", item)
    cfg_flow_other = get_json_file(cfg_dir_flow_other)
    inds_df1, pred, obs = stat_result(
        cfg_flow_other["data_params"]["test_path"],
        cfg_flow_other["evaluate_params"]["test_epoch"],
        fill_nan=cfg_flow_other["evaluate_params"]["fill_nan"],
        var_unit=[
            hydro_constant.streamflow.unit,
            hydro_constant.surface_soil_moisture.unit,
        ],
        var_name=[
            hydro_constant.streamflow.name,
            hydro_constant.surface_soil_moisture.name,
        ],
    )
    inds_all_lst.append(inds_df1[0]["NSE"].values)
plot_ecdf_func(
    inds_all_lst,
    ["PUB-stlq", "PUB-mtl"],
    save_path=os.path.join(cfg_dir_flow_other, "pub_nse_comp.png"),
)
