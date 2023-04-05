"""
Author: Wenyu Ouyang
Date: 2022-07-22 18:06:00
LastEditTime: 2022-11-10 17:33:37
LastEditors: Wenyu Ouyang
Description: Plots for better predictions of streamflow
FilePath: /HydroSPB/hydroSPB/app/multi_task/mtl_better_streamflow.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from matplotlib import pyplot as plt
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydroSPB.app.multi_task.mtl_results_utils import (
    plot_multi_single_comp_flow_boxes,
    read_multi_single_exps_results,
)
from hydroSPB.visual.plot_stat import plot_boxes_matplotlib, plot_scatter_with_11line

cases_exps_legends_together = [
    "0",
    "1/24",
    "1/8",
    "1/3",
    "1",
    # "1/2",
    # "1/5",
    # "3",
    # "8",
    # "mean",
    "best",
]
show_inds = ["Bias", "RMSE", "Corr", "KGE"]

# mtl_q_et_valid_exps = ["exp4248", "exp4249", "exp4250", "exp4251"]
# mtl_q_et_test_exps = ["exp4270", "exp4271", "exp4272", "exp4273"]
mtl_q_et_valid_exps = ["exp41031", "exp41025", "exp41019", "exp41013"]
mtl_q_et_test_exps = [tmp + "0" for tmp in mtl_q_et_valid_exps]
# stl_q_valid_exps = ["exp4247"]
# stl_q_test_exps = ["exp4247"]
stl_q_valid_exps = ["exp41001"]
stl_q_test_exps = ["exp410010"]
exps_q_et_valid = stl_q_valid_exps + mtl_q_et_valid_exps
exps_q_et_test = stl_q_test_exps + mtl_q_et_test_exps
exps_q_et_results_valid, q_et_best_index_valid = read_multi_single_exps_results(
    exps_q_et_valid, ensemble=-1
)
exps_q_et_results, _ = read_multi_single_exps_results(
    exps_q_et_test, q_et_best_index_valid, ensemble=-1
)
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

# plot scatter with a 1:1 line to compare single-task and multi-task models
plot_scatter_with_11line(
    exps_q_et_results[0],
    exps_q_et_results[-1],
    # xlabel="NSE single-task",
    # ylabel="NSE multi-task",
    xlabel="径流单任务学习模型预测NSE",
    ylabel="多任务学习模型径流预测NSE",
)
plt.savefig(
    os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        exps_q_et_test[-1],
        "mtl_stl_scatter_plot_with_11line.png",
    ),
    dpi=600,
    bbox_inches="tight",
)

# plot boxes of NSEs for multi plans
plot_multi_single_comp_flow_boxes(
    exps_q_et_results,
    cases_exps_legends_together=cases_exps_legends_together,
    save_path=os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        exps_q_et_test[-1],
        "mtl_syn_flow_et_boxes.png",
    ),
    rotation=45,
)


# plot other metrics for q-et
exps_q_et_metrices_results = []
for ind in show_inds:
    exps_q_et_metric_results, _ = read_multi_single_exps_results(
        exps_q_et_test, q_et_best_index_valid, metric=ind
    )
    exps_q_et_metrices_results.append(
        [exps_q_et_metric_results[0], exps_q_et_metric_results[-1]]
    )
# https://www.statology.org/change-font-size-matplotlib/
plt.rc("axes", labelsize=16)
plt.rc("ytick", labelsize=12)
FIGURE_DPI = 600
plot_boxes_matplotlib(
    exps_q_et_metrices_results,
    label1=show_inds,
    label2=["λ=0", "λ=best"],
    colorlst=["#d62728", "#1f77b4"],
    figsize=(10, 5),
    subplots_adjust_wspace=0.35,
    median_font_size="xx-small",
)
plt.savefig(
    os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        exps_q_et_test[-1],
        "mtl_syn_flow_et_metrices_boxes.png",
    ),
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

mtl_q_ssm_valid_exps = [
    "exp41055",
    "exp41049",
    "exp41043",
    "exp41037",
    # "exp41200",
    # "exp41206",
    # "exp41224",
    # "exp41230",
]
mtl_q_ssm_test_exps = [tmp + "0" for tmp in mtl_q_ssm_valid_exps]
# mtl_q_ssm_valid_exps = ["exp4235", "exp4236", "exp4237", "exp4238"]
# mtl_q_ssm_test_exps = ["exp4241", "exp4242", "exp4243", "exp4244"]
# stl_q_15_21_valid_exps = ["exp4234"]
# stl_q_15_21_test_exps = ["exp4234"]
stl_q_15_21_valid_exps = ["exp41007"]
stl_q_15_21_test_exps = ["exp410070"]
exps_q_ssm_valid = stl_q_15_21_valid_exps + mtl_q_ssm_valid_exps
exps_q_ssm_test = stl_q_15_21_test_exps + mtl_q_ssm_test_exps
exps_q_ssm_results_valid, q_ssm_best_index_valid = read_multi_single_exps_results(
    exps_q_ssm_valid
)
exps_q_ssm_nse_results, best_index = read_multi_single_exps_results(
    exps_q_ssm_test, q_ssm_best_index_valid
)
# plot boxes for NSE
plot_multi_single_comp_flow_boxes(
    exps_q_ssm_nse_results,
    cases_exps_legends_together=cases_exps_legends_together,
    save_path=os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        exps_q_ssm_test[-1],
        "mtl_syn_flow_ssm_boxes.png",
    ),
    rotation=45,
)

# plot other metrices for q-ssm
exps_q_ssm_metrices_results = []
for ind in show_inds:
    exps_q_ssm_metric_results, _ = read_multi_single_exps_results(
        exps_q_ssm_test, q_ssm_best_index_valid, metric=ind
    )
    exps_q_ssm_metrices_results.append(
        [exps_q_ssm_metric_results[0], exps_q_ssm_metric_results[-1]]
    )

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
# https://www.statology.org/change-font-size-matplotlib/
plt.rc("axes", labelsize=16)
plt.rc("ytick", labelsize=12)
# plot multiple metrices' boxes in one plot
plot_boxes_matplotlib(
    exps_q_ssm_metrices_results,
    label1=show_inds,
    label2=["λ=0", "λ=best"],
    colorlst=["#d62728", "#1f77b4"],
    figsize=(10, 5),
    subplots_adjust_wspace=0.35,
    median_font_size="xx-small",
)
FIGURE_DPI = 600
plt.savefig(
    os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        exps_q_ssm_test[-1],
        "mtl_syn_flow_ssm_metrices_boxes.png",
    ),
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)
