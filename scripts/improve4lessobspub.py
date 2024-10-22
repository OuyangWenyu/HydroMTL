"""
Author: Wenyu Ouyang
Date: 2024-10-14 20:34:23
LastEditTime: 2024-10-22 10:39:20
LastEditors: Wenyu Ouyang
Description: evaluate the results
FilePath: \HydroMTL\scripts\improve4lessobspub.py
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
from hydromtl.utils.hydro_stat import stat_error
from hydromtl.visual.plot_stat import plot_scatter_with_11line
from scripts.streamflow_utils import get_json_file, plot_ecdf_func

# we have 2-fold results to see, one-fold for some basins and the other for the rest
# so we directly compare all basins results
exps_lst = [
    [
        [
            "exppubstlq8010",
            "exppubstlq8030",
            "exppubstlq8050",
            "exppubstlq8070",
            "exppubstlq8090",
        ],
        [
            "exppubstlq8020",
            "exppubstlq8040",
            "exppubstlq8060",
            "exppubstlq8080",
            "exppubstlq8100",
        ],
    ],
    [
        [
            "exppubmtl7010",
            "exppubmtl7030",
            "exppubmtl7050",
            "exppubmtl7070",
            "exppubmtl7090",
        ],
        [
            "exppubmtl7020",
            "exppubmtl7040",
            "exppubmtl7060",
            "exppubmtl7080",
            "exppubmtl7100",
        ],
    ],
]
inds_all_lst = []
var_idx = 0
for models_exp in exps_lst:
    pred_all_fold = []
    obs_all_fold = []
    for ensemble_exps in models_exp:
        preds = []
        obss = []
        for exp in ensemble_exps:
            cfg_dir_flow_other = os.path.join(definitions.RESULT_DIR, "camels", exp)
            cfg_flow_other = get_json_file(cfg_dir_flow_other)
            inds_df, pred, obs = stat_result(
                cfg_flow_other["data_params"]["test_path"],
                cfg_flow_other["evaluate_params"]["test_epoch"],
                fill_nan=cfg_flow_other["evaluate_params"]["fill_nan"],
                var_unit=[hydro_constant.streamflow.unit],
                var_name=[hydro_constant.streamflow.name],
                return_value=True,
            )
            preds.append(pred[var_idx])
            obss.append(obs[var_idx])
        pred_ensemble = np.array(preds).mean(axis=0)
        obs_ensemble = np.array(obss).mean(axis=0)
        pred_all_fold.append(pred_ensemble)
        obs_all_fold.append(obs_ensemble)
    preds_all_fold = np.concatenate(pred_all_fold)
    obss_all_fold = np.concatenate(obs_all_fold)
    inds_all = stat_error(
        obss_all_fold,
        preds_all_fold,
        fill_nan="no",
    )
    inds_all_lst.append(inds_all["NSE"])
cases_exps_legends_together = ["STL_Q_PUB", "MTL_PUB"]
for i in range(len(cases_exps_legends_together)):
    print(
        f"the median NSE of {cases_exps_legends_together[i]} is {np.median(inds_all_lst[i])}"
    )
figure_dir = os.path.join(definitions.RESULT_DIR, "figures", "data_augment")
plot_ecdf_func(
    inds_all_lst,
    cases_exps_legends_together,
    save_path=os.path.join(figure_dir, "mtl_stl_pub_nse_comp.png"),
)
_, _, textstr = plot_scatter_with_11line(
    inds_all_lst[0],
    inds_all_lst[1],
    # xlabel="NSE single-task",
    # ylabel="NSE multi-task",
    xlabel=f"{cases_exps_legends_together[0]} NSE",
    ylabel=f"{cases_exps_legends_together[1]} NSE",
)
print(textstr)
plt.savefig(
    os.path.join(
        figure_dir,
        "mtl_stl_pub_q_scatter_plot_with_11line.png",
    ),
    dpi=600,
    bbox_inches="tight",
)
