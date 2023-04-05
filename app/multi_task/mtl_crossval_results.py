"""
Author: Wenyu Ouyang
Date: 2022-05-04 17:56:14
LastEditTime: 2022-12-16 16:52:54
LastEditors: Wenyu Ouyang
Description: Plot for Cross Validation exps
FilePath: /HydroSPB/hydroSPB/app/multi_task/mtl_crossval_results.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from functools import reduce
import sys

import os
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydroSPB.app.streamflow_utils import (
    get_json_file,
    get_lastest_weight_path,
    plot_ecdf_func,
)
from hydroSPB.hydroDL.trainer import stat_result
from hydroSPB.utils.hydro_stat import stat_error
from hydroSPB.app.multi_task.mtl_results_utils import (
    predict_new_et_exp,
    predict_new_mtl_exp,
    predict_new_q_exp,
)
from hydroSPB.data.source.data_constant import ET_MODIS_NAME, Q_CAMELS_US_NAME

split_num = 5
# test in trained basins for test period
mtl_q_train_exps = ["exp4300" + str(i + 1) for i in range(split_num)]
mtl_q_et_train_exps = [
    ["exp" + str(i) for i in range(43007, 43007 + split_num)],
    ["exp" + str(i) for i in range(43013, 43013 + split_num)],
    ["exp" + str(i) for i in range(43019, 43019 + split_num)],
    ["exp" + str(i) for i in range(43025, 43025 + split_num)],
]
mtl_et_train_exps = ["exp4303" + str(i + 1) for i in range(split_num)]

# test in tested basins for test period
mtl_q_test_exps = [tmp + "0" for tmp in mtl_q_train_exps]
mtl_q_et_test_exps = [[i + "0" for i in tmp] for tmp in mtl_q_et_train_exps]
mtl_et_test_exps = [tmp + "0" for tmp in mtl_et_train_exps]

loss_weights = [[0.5, 0.5], [0.75, 0.25], [0.88, 0.11], [0.96, 0.04]]
test_epoch = 300
FIGURE_DPI = 600

test = False
if test:
    # pub test
    cache_path = [None] * split_num
    cache_path_valid = [None] * split_num
    gages_id_files = [
        os.path.join(
            definitions.ROOT_DIR,
            "hydroSPB",
            "example",
            "camels",
            "exp43_kfold" + str(split_num),
            "camels_test_kfold" + str(tmp) + ".csv",
        )
        for tmp in range(split_num)
    ]
    # test for et
    weight_path_et_dir = [
        os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", "camels", exp)
        for exp in mtl_et_train_exps
    ]
    weight_path_et = [get_lastest_weight_path(dir_) for dir_ in weight_path_et_dir]
    stat_dict_file_et = [
        os.path.join(dir_, "dapengscaler_stat.json") for dir_ in weight_path_et_dir
    ]
    for i in range(len(mtl_et_train_exps)):
        predict_new_et_exp(
            exp=mtl_et_test_exps[i],
            weight_path=weight_path_et[i],
            train_period=["2001-10-01", "2011-10-01"],
            test_period=["2011-10-01", "2021-10-01"],
            cache_path=cache_path[i],
            gage_id_file=gages_id_files[i],
            stat_dict_file=stat_dict_file_et[i],
        )
    # test for q
    weight_path_q_dir = [
        os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", "camels", exp)
        for exp in mtl_q_train_exps
    ]
    weight_path_q = [get_lastest_weight_path(dir_) for dir_ in weight_path_q_dir]
    stat_dict_file_q = [
        os.path.join(dir_, "dapengscaler_stat.json") for dir_ in weight_path_q_dir
    ]
    for j in range(len(mtl_q_train_exps)):
        predict_new_q_exp(
            exp=mtl_q_test_exps[j],
            weight_path=weight_path_q[j],
            train_period=["2001-10-01", "2011-10-01"],
            test_period=["2011-10-01", "2021-10-01"],
            cache_path=cache_path[j],
            gage_id_file=gages_id_files[j],
            stat_dict_file=stat_dict_file_q[j],
        )
    # test for q-et
    count = 0
    for mtl_exp in mtl_q_et_test_exps:
        mtl_train_exp = mtl_q_et_train_exps[count]
        weight_path_q_et_dir = [
            os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", "camels", exp)
            for exp in mtl_train_exp
        ]
        weight_path_q_et = [
            get_lastest_weight_path(dir_) for dir_ in weight_path_q_et_dir
        ]
        stat_dict_file_q_et = [
            os.path.join(dir_, "dapengscaler_stat.json")
            for dir_ in weight_path_q_et_dir
        ]
        if count > 0:
            cache_path = [
                os.path.join(
                    definitions.ROOT_DIR,
                    "hydroSPB",
                    "example",
                    "camels",
                    mtl_q_et_test_exps[0][k],
                )
                for k in range(split_num)
            ]
            cache_path_valid = [
                os.path.join(
                    definitions.ROOT_DIR,
                    "hydroSPB",
                    "example",
                    "camels",
                    mtl_q_et_test_exps[0][k],
                )
                for k in range(split_num)
            ]
        for i in range(split_num):
            predict_new_mtl_exp(
                exp=mtl_q_et_test_exps[count][i],
                targets=[Q_CAMELS_US_NAME, ET_MODIS_NAME],
                loss_weights=loss_weights[count],
                weight_path=weight_path_q_et[i],
                train_period=["2001-10-01", "2011-10-01"],
                test_period=["2011-10-01", "2021-10-01"],
                cache_path=cache_path[i],
                gage_id_file=gages_id_files[i],
                stat_dict_file=stat_dict_file_q_et[i],
            )
        count += 1


def stat_stl_mtl_ensemble_result(
    exps,
    test_epoch,
    return_value=False,
    mtl=True,
    # set unit of ET as mm_day  rather than mm/day, because mm/day is set for streamflow and it will be transferred to m3/s
    units=["ft3/s", "mm_day"],
    fill_nan=["no", "mean"],
    idx_in_mtl=0,
):
    preds = []
    obss = []
    inds = []
    for i in range(len(exps)):
        cfg_dir = os.path.join(
            definitions.ROOT_DIR, "hydroSPB", "example", "camels", exps[i]
        )
        cfg_ = get_json_file(cfg_dir)
        # TODO: need check because we refactor stat_result
        _, pred_i, obs_i = stat_result(
            cfg_["data_params"]["test_path"],
            test_epoch,
            return_value=True,
            unit=units,
            fill_nan=fill_nan,
        )
        if mtl:
            preds.append(pred_i[:, :, idx_in_mtl])
            obss.append(obs_i[:, :, idx_in_mtl])
            inds_i = stat_error(
                obs_i[:, :, idx_in_mtl],
                pred_i[:, :, idx_in_mtl],
                fill_nan=fill_nan[idx_in_mtl],
            )
        else:
            preds.append(pred_i)
            obss.append(obs_i)
            inds_i = stat_error(obs_i, pred_i)
        inds.append(inds_i)
    preds_np = reduce(lambda a, b: np.vstack((a, b)), preds)
    obss_np = reduce(lambda a, b: np.vstack((a, b)), obss)
    inds_ = stat_error(obss_np, preds_np)
    inds_df = pd.DataFrame(inds_)
    if return_value:
        return inds_df, preds_np, obss_np, inds
    return inds_df


def which_is_best(
    inds_all_lst,
    best_valid_idx=None,
    ensemble=False,
    preds=None,
    obss=None,
    streamflow=True,
    metric=None,
):
    # get mean but not ensemble
    if ensemble:
        mtl_mean_results = []
        pred_ensemble = np.array(preds).mean(axis=0)
        obs_ensemble = np.array(obss).mean(axis=0)
        if streamflow:
            fill_nan = "no"
        else:
            fill_nan = "mean"
        inds_ensemble = stat_error(obs_ensemble, pred_ensemble, fill_nan=fill_nan)
        mtl_mean_results = pd.DataFrame(inds_ensemble)[metric].values
    else:
        mtl_mean_results = np.array(inds_all_lst).mean(axis=0)
    if best_valid_idx is None:
        mtl_best_results = np.array(inds_all_lst).max(axis=0)
        mtl_best_results_where = np.array(inds_all_lst).argmax(axis=0)
    else:
        mtl_best_results_where = best_valid_idx
        mtl_results = np.array(inds_all_lst)
        mtl_best_results = np.array(
            [mtl_results[idx, i] for i, idx in enumerate(best_valid_idx)]
        )
    inds_all_lst.append(mtl_mean_results)
    inds_all_lst.append(mtl_best_results)
    return inds_all_lst, mtl_best_results_where


def concat_mtl_stl_result(
    train_test_exps,
    test_test_exps,
    stl_train_exps,
    stl_test_exps,
    ind_names,
    for_flow=True,
):
    """concatenate MTL's results with Single-task-learning's result for trained basins and pub test basins
    train_test_exps
        mtl testing results of trained basins
    test_test_exps
        mtl testing results of tested basins
    stl_train_exps
        stl testing results of trained basins
    stl_test_exps
        stl testing results of tested basins
    """
    keys_results = []
    exps_mtl_results_train_tests = []
    exps_mtl_results_test_tests = []
    exps_mtl_results_train_tests_preds = []
    exps_mtl_results_train_tests_obss = []
    exps_mtl_results_test_tests_preds = []
    exps_mtl_results_test_tests_obss = []
    for _ in ind_names:
        exps_mtl_results_train_tests.append([])
        exps_mtl_results_test_tests.append([])
    if for_flow:
        idx_in_mtl = 0
        stl_unit = "ft3/s"
        stl_fill_nan = "no"
    else:
        idx_in_mtl = 1
        # mm_day is specially for ET
        stl_unit = "mm_day"
        stl_fill_nan = "mean"
    for mtl_train_test_exps in train_test_exps:
        (
            inds_df_mtl_train_test,
            pred_mean_mtl_train_test,
            obs_mean_mtl_train_test,
            all_inds_mtl_train_test,
        ) = stat_stl_mtl_ensemble_result(
            mtl_train_test_exps, test_epoch, return_value=True, idx_in_mtl=idx_in_mtl
        )
        for i in range(len(ind_names)):
            exps_mtl_results_train_tests[i].append(
                inds_df_mtl_train_test[ind_names[i]].values
            )
        exps_mtl_results_train_tests_preds.append(pred_mean_mtl_train_test)
        exps_mtl_results_train_tests_obss.append(obs_mean_mtl_train_test)
    for mtl_test_test_exps in test_test_exps:
        (
            inds_df_mtl_test_test,
            pred_mean_mtl_test_test,
            obs_mean_mtl_test_test,
            all_inds_mtl_test_test,
        ) = stat_stl_mtl_ensemble_result(
            mtl_test_test_exps, test_epoch, return_value=True, idx_in_mtl=idx_in_mtl
        )
        for i in range(len(ind_names)):
            exps_mtl_results_test_tests[i].append(
                inds_df_mtl_test_test[ind_names[i]].values
            )
        exps_mtl_results_test_tests_preds.append(pred_mean_mtl_test_test)
        exps_mtl_results_test_tests_obss.append(obs_mean_mtl_test_test)
    for i in range(len(ind_names)):
        exps_mtl_results_train_tests[i], best_index = which_is_best(
            exps_mtl_results_train_tests[i]
        )
        exps_mtl_results_test_tests[i], best_index = which_is_best(
            exps_mtl_results_test_tests[i]
        )

    (
        inds_df_q_train,
        pred_mean_q_train,
        obs_mean_q_train,
        all_inds_q_train,
    ) = stat_stl_mtl_ensemble_result(
        stl_train_exps,
        test_epoch,
        return_value=True,
        mtl=False,
        units=stl_unit,
        fill_nan=stl_fill_nan,
    )
    (
        inds_df_q_test,
        pred_mean_q_test,
        obs_mean_q_test,
        all_inds_q_test,
    ) = stat_stl_mtl_ensemble_result(
        stl_test_exps,
        test_epoch,
        return_value=True,
        mtl=False,
        units=stl_unit,
        fill_nan=stl_fill_nan,
    )
    for i in range(len(ind_names)):
        exps_mtl_results_train_tests[i].append(inds_df_q_train[ind_names[i]].values)
        exps_mtl_results_test_tests[i].append(inds_df_q_test[ind_names[i]].values)
    for i in range(len(ind_names)):
        # get best
        # keys_results.append(
        #     exps_mtl_results_train_tests[i][-2:] + exps_mtl_results_test_tests[i][-2:]
        # )
        # get mean
        keys_results.append(
            exps_mtl_results_train_tests[i][-3:-2]
            + exps_mtl_results_train_tests[i][-1:]
            + exps_mtl_results_test_tests[i][-3:-2]
            + exps_mtl_results_test_tests[i][-1:]
        )
        # get best for valid and mean for test
        # keys_results.append(
        #     exps_mtl_results_train_tests[i][-2:]
        #     + exps_mtl_results_test_tests[i][-3:-2]
        #     + exps_mtl_results_test_tests[i][-1:]
        # )
    return keys_results


# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


# plot for ET
# cases_exps_legends_together = [
#     "MTL_ET",
#     "ET",
#     "MTL_ET_PUB",
#     "ET_PUB",
# ]
cases_exps_legends_together = [
    "多任务学习蒸散发",
    "单任务学习蒸散发",
    "多任务学习PUB蒸散发",
    "单任务学习PUB蒸散发",
]
keys_ecdf = ["Bias", "Corr", "NSE", "KGE", "FHV", "FLV"]
x_intervals = [1, 0.1, 0.1, 0.1, 20, 20]
x_lims = [(-5, 5), (0, 1), (0, 1), (0, 1), (-50, 150), (-50, 150)]
key_results = concat_mtl_stl_result(
    mtl_q_et_train_exps,
    mtl_q_et_test_exps,
    mtl_et_train_exps,
    mtl_et_test_exps,
    keys_ecdf,
    for_flow=False,
)
idx_tmp = 0
for key_tmp in keys_ecdf:
    key_result = key_results[idx_tmp]
    plot_ecdf_func(
        key_result,
        cases_exps_legends_together=cases_exps_legends_together,
        save_path=os.path.join(
            definitions.ROOT_DIR,
            "hydroSPB",
            "example",
            "camels",
            mtl_q_et_test_exps[-1][-1],
            "mtl_cross_val_et_" + key_tmp + "_ecdf.png",
        ),
        dash_lines=[False, True, False, True],
        colors="rrbb",
        x_interval=x_intervals[idx_tmp],
        x_lim=x_lims[idx_tmp],
        x_str=key_tmp,
        show_legend=True,
        legend_font_size=15
        # show_legend=True if idx_tmp == 0 else False,
    )
    idx_tmp = idx_tmp + 1

# plot for Q
# cases_exps_legends_together = [
#     "MTL_Q",
#     "Q",
#     "MTL_Q_PUB",
#     "Q_PUB",
# ]
cases_exps_legends_together = [
    "多任务学习径流",
    "单任务学习径流",
    "多任务学习PUB径流",
    "单任务学习PUB径流",
]
keys_ecdf = ["Bias", "Corr", "NSE", "KGE", "FHV", "FLV"]
x_intervals = [2, 0.1, 0.1, 0.1, 50, 50]
x_lims = [(-10, 10), (0, 1), (0, 1), (0, 1), (-100, 300), (-100, 300)]
key_results = concat_mtl_stl_result(
    mtl_q_et_train_exps,
    mtl_q_et_test_exps,
    mtl_q_train_exps,
    mtl_q_test_exps,
    keys_ecdf,
)
idx_tmp = 0

for key_tmp in keys_ecdf:
    key_result = key_results[idx_tmp]
    plot_ecdf_func(
        key_result,
        cases_exps_legends_together=cases_exps_legends_together,
        save_path=os.path.join(
            definitions.ROOT_DIR,
            "hydroSPB",
            "example",
            "camels",
            mtl_q_et_test_exps[-1][-1],
            "mtl_cross_val_q_" + key_tmp + "_ecdf.png",
        ),
        dash_lines=[False, True, False, True],
        colors="rrbb",
        x_interval=x_intervals[idx_tmp],
        x_lim=x_lims[idx_tmp],
        x_str=key_tmp,
        show_legend=True,
        legend_font_size=15
        # show_legend=True if idx_tmp == 0 else False,
    )
    idx_tmp = idx_tmp + 1
