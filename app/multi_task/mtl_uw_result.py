"""
Author: Wenyu Ouyang
Date: 2022-04-02 09:37:47
LastEditTime: 2023-02-11 08:22:44
LastEditors: Wenyu Ouyang
Description: Plots for MTL data augmentation exps
FilePath: /HydroSPB/hydroSPB/app/multi_task/mtl_uw_result.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import sys
from pathlib import Path
from matplotlib import pyplot as plt

import pandas as pd


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydroSPB.app.streamflow_utils import get_json_file, plot_ecdf_func
from hydroSPB.hydroDL.trainer import stat_result, train_and_evaluate
from hydroSPB.visual.plot_stat import plot_boxes_matplotlib
from hydroSPB.app.multi_task.mtl_results_utils import (
    plot_uw_sigma,
    read_multi_single_exps_results,
)

result_dir = os.path.join(
    definitions.ROOT_DIR,
    "hydroSPB",
    "app",
    "multi_task",
    "results",
)
cases_exps_legends_together = [
    "λ=0",
    "UW",
]
exps_et = ["exp42001", "exp41701"]
exps_et_q_results = read_multi_single_exps_results(
    exps_et, var_idx=1, single_is_flow=False
)
plot_ecdf_func(
    exps_et_q_results,
    cases_exps_legends_together=cases_exps_legends_together,
    save_path=os.path.join(
        result_dir,
        "mtl_syn_et_flow_uw_ecdf_" + exps_et[1] + ".png",
    ),
)

exps_flow = ["exp41001", "exp41701"]
exps_q_et_results = read_multi_single_exps_results(exps_flow)
plot_ecdf_func(
    exps_q_et_results,
    cases_exps_legends_together=cases_exps_legends_together,
    save_path=os.path.join(
        definitions.ROOT_DIR,
        result_dir,
        "mtl_syn_flow_et_uw_ecdf_" + exps_et[1] + ".png",
    ),
)


exps_sigma = ["exp41701"]
plot_uw_sigma(exps_sigma, [["Q", "ET"]], ["uw_sigma_q_et.png"], result_dir=result_dir)


def show_mtl_uw_sigma(exp):
    """TODO: Show sigma values of uncertainty weights

    Parameters
    ----------
    exp : str
        the name of exp
    """
    cfg_dir = os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", "camels", exp)
    cfg = get_json_file(cfg_dir)
    epoch = cfg["evaluate_params"]["test_epoch"]
    weight_path = os.path.join(cfg_dir, "model_Ep" + str(epoch) + ".pth")
    cfg["model_params"]["weight_path"] = weight_path
    if cfg["data_params"]["cache_read"]:
        cfg["data_params"]["cache_write"] = False
    cfg["training_params"]["train_mode"] = False
    train_and_evaluate(cfg)
    print("See weights of trained model")


def comp_mtl_aug_cases(
    exps, leg_names, colors, fig_size, fig_name, subplots_adjust_wspace
):
    inds_df_lst = []
    for exp in exps:
        cfg_dir = os.path.join(
            definitions.ROOT_DIR, "hydroSPB", "example", "camels", exp
        )
        cfg_dict = get_json_file(cfg_dir)
        inds_df, pred, obs = stat_result(
            cfg_dict["data_params"]["test_path"],
            cfg_dict["evaluate_params"]["test_epoch"],
            fill_nan=cfg_dict["evaluate_params"]["fill_nan"],
            return_value=True,
        )
        if type(inds_df) is list:
            # two outputs
            inds_df_lst.append(inds_df[1])
        else:
            inds_df_lst.append(inds_df)
    show_inds = ["Bias", "RMSE", "Corr", "NSE"]
    concat_inds = [
        [df[ind].values if type(df) is pd.DataFrame else df[ind] for df in inds_df_lst]
        for ind in show_inds
    ]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    # https://www.statology.org/change-font-size-matplotlib/
    plt.rc("axes", labelsize=16)
    plt.rc("ytick", labelsize=12)
    FIGURE_DPI = 600
    plot_boxes_matplotlib(
        concat_inds,
        label1=show_inds,
        label2=leg_names,
        colorlst=colors,
        figsize=fig_size,
        subplots_adjust_wspace=subplots_adjust_wspace,
    )
    plt.savefig(
        os.path.join(cfg_dir, fig_name),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )


# show_mtl_uw_sigma("exp414")
# comp_mtl_aug_cases(
#     ["exp415", "exp417", "exp4111"],
#     ["SSM", "SSM_MTL", "SSM_MTL_PRO"],
#     ["#d62728", "#1f77b4", "#2ca02c"],
#     (12, 6),
#     "mtl_aug_ssm_ssm_mtl_comp.png",
#     subplots_adjust_wspace=0.3,
# )
comp_mtl_aug_cases(
    ["exp42001", "exp41701"],
    ["ET", "ET_MTL"],
    ["#d62728", "#1f77b4"],
    (10, 5),
    "mtl_aug_et_et_mtl_comp.png",
    subplots_adjust_wspace=0.35,
)
