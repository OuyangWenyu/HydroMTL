"""
Author: Wenyu Ouyang
Date: 2022-07-25 22:02:43
LastEditTime: 2022-08-17 21:56:37
LastEditors: Wenyu Ouyang
Description: Show the results of 3-output models for enhancing streamflow prediction
FilePath: /HydroSPB/hydroSPB/app/multi_task/mtl_3task_results.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import numpy as np
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydroSPB.app.multi_task.mtl_results_utils import (
    plot_mtl_results_map,
    plot_multi_single_comp_flow_boxes,
    read_multi_single_exps_results,
)
from hydroSPB.visual.plot_stat import plot_heat_map
from hydroSPB.utils.hydro_stat import wilcoxon_t_test_for_lst

# weights: 1:1:1, 3:1:1, 8:1:1, 24:1:1
# mtl_q_et_ssm_valid_exps = ["exp4254", "exp4255", "exp4256", "exp4257"]
# mtl_q_et_ssm_test_exps = ["exp4258", "exp4259", "exp4260", "exp4261"]
mtl_q_et_ssm_valid_exps = ["exp41085", "exp41091", "exp41097", "exp41103"]
mtl_q_et_ssm_test_exps = [tmp + "0" for tmp in mtl_q_et_ssm_valid_exps]
# mtl_q_ssm_valid_exps = ["exp4235", "exp4236", "exp4237", "exp4238"]
# mtl_q_ssm_test_exps = ["exp4241", "exp4242", "exp4243", "exp4244"]
mtl_q_ssm_valid_exps = ["exp41037", "exp41043", "exp41049", "exp41055"]
mtl_q_ssm_test_exps = [tmp + "0" for tmp in mtl_q_ssm_valid_exps]
# mtl_q_et_15_21_valid_exps = ["exp4274", "exp4275", "exp4276", "exp4277"]
# mtl_q_et_15_21_test_exps = ["exp4278", "exp4279", "exp4280", "exp4281"]
mtl_q_et_15_21_valid_exps = ["exp41061", "exp41067", "exp41073", "exp41079"]
mtl_q_et_15_21_test_exps = [tmp + "0" for tmp in mtl_q_et_15_21_valid_exps]
# stl_q_15_21_valid_exps = ["exp4234"]
# stl_q_15_21_test_exps = ["exp4234"]
stl_q_15_21_valid_exps = ["exp41007"]
stl_q_15_21_test_exps = ["exp410070"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

cases_exps_legends_together = ["0", "1", "1/3", "1/8", "1/24", "mean", "best"]

exps_valid3 = stl_q_15_21_valid_exps + mtl_q_et_ssm_valid_exps
exps_test3 = stl_q_15_21_test_exps + mtl_q_et_ssm_test_exps
exps_results_valid3, best_index_valid3 = read_multi_single_exps_results(exps_valid3)
exps_results3, best_index3 = read_multi_single_exps_results(
    exps_test3, best_index_valid3
)

exps_q_ssm_valid = stl_q_15_21_valid_exps + mtl_q_ssm_valid_exps
exps_q_ssm_test = stl_q_15_21_test_exps + mtl_q_ssm_test_exps
exps_q_ssm_results_valid, q_ssm_best_index_valid = read_multi_single_exps_results(
    exps_q_ssm_valid
)
exps_q_ssm_results, best_index = read_multi_single_exps_results(
    exps_q_ssm_test, q_ssm_best_index_valid
)

exps_q_et_15_21_valid = stl_q_15_21_valid_exps + mtl_q_et_15_21_valid_exps
exps_q_et_15_21_test = stl_q_15_21_test_exps + mtl_q_et_15_21_test_exps
exps_q_et_15_21_results_valid, best_15_21_index_valid = read_multi_single_exps_results(
    exps_q_et_15_21_valid
)
exps_q_et_15_21_results, best_15_21_index_test = read_multi_single_exps_results(
    exps_q_et_15_21_test, best_15_21_index_valid
)


all_mtl_valid_exps = (
    stl_q_15_21_valid_exps
    + mtl_q_et_15_21_valid_exps
    + mtl_q_ssm_valid_exps
    + mtl_q_et_ssm_valid_exps
)
all_mtl_test_exps = (
    stl_q_15_21_test_exps
    + mtl_q_et_15_21_test_exps
    + mtl_q_ssm_test_exps
    + mtl_q_et_ssm_test_exps
)
all_mtl_exp_valid_results, all_mtl_exps_valid_indices = read_multi_single_exps_results(
    all_mtl_valid_exps
)
all_mtl_exp_test_results, all_mtl_exps_test_indices = read_multi_single_exps_results(
    all_mtl_test_exps, all_mtl_exps_valid_indices
)

exps_best_results_allcases = np.array(
    [
        exps_results3[0],
        exps_q_et_15_21_results[-1],
        exps_q_ssm_results[-1],
        exps_results3[-1],
        all_mtl_exp_test_results[-1],
    ]
)
# plot best cases in q, q-et, q-ssm, q-et-smm, all mtl tasks
casesall_best_exps = ["Q", "Q-ET", "Q-SSM", "Q-ET-SSM", "ALL_MTL"]
plot_multi_single_comp_flow_boxes(
    exps_best_results_allcases,
    cases_exps_legends_together=casesall_best_exps,
    save_path=os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        exps_test3[-1],
        "mtl_syn_best_cases_boxes_allcases.png",
    ),
    rotation=45,
    x_name="expriment",
)

exps_best_results = np.array(
    [
        exps_results3[0],
        exps_q_et_15_21_results[-1],
        exps_q_ssm_results[-1],
        exps_results3[-1],
    ]
)
cases_best_exps = ["Q", "Q-ET", "Q-SSM", "Q-ET-SSM"]
plot_multi_single_comp_flow_boxes(
    exps_best_results,
    cases_exps_legends_together=cases_best_exps,
    save_path=os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        exps_test3[-1],
        "mtl_syn_best_cases_boxes.png",
    ),
    rotation=45,
    x_name="expriment",
)

plot_mtl_results_map(
    exps_best_results_allcases[[0, -1]],
    ["Q", "ALL_MTL"],
    ["o", "x"],
    os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        exps_test3[-1],
        "best_stl_mtl_cases_map.png",
    ),
)

plot_mtl_results_map(
    exps_best_results,
    ["Q", "Q-ET", "Q-SSM", "Q-ET-SSM"],
    ["o", "v", "+", "x"],
    os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        exps_test3[-1],
        "best_cases_map.png",
    ),
)
w, p = wilcoxon_t_test_for_lst(exps_best_results[[0, -1]])
row_num, col_num = int((len(all_mtl_valid_exps) - 1) / 3), 3
data4heatmap = np.zeros([row_num, col_num])
for i in range(col_num):
    for j in range(row_num):
        data4heatmap[j, i] = np.sum(all_mtl_exps_valid_indices == i * row_num + j)
df4heatmap = pd.DataFrame(
    data4heatmap,
    columns=["Q-ET", "Q-SSM", "Q-ET-SSM"],
    index=["1", "1/3", "1/8", "1/24"],
    dtype=int,
)
plot_heat_map(df4heatmap, fig_size=(4, 4))
plt.savefig(
    os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        exps_test3[-1],
        "best_cases_heatmap.png",
    ),
    dpi=600,
    bbox_inches="tight",
)
