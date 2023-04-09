"""
Author: Wenyu Ouyang
Date: 2022-07-24 14:45:05
LastEditTime: 2023-04-09 21:50:40
LastEditors: Wenyu Ouyang
Description: Plots for MTL valid and test results comparing with STL
FilePath: /HydroMTL/scripts/mtl_better_prediction.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from mtl_results_utils import (
    plot_mtl_results_map,
    plot_multi_metrics_for_stl_mtl,
    plot_multi_single_comp_flow_boxes,
    read_multi_single_exps_results,
)
from hydromtl.visual.plot_stat import plot_scatter_with_11line, plot_ts
from hydromtl.utils import hydro_utils


def plot_for_prediction_performance(args):
    mtl_exps = args.mtl_exps
    stl_q_exps = args.stl_q_exps
    stl_et_exps = args.stl_et_exps
    mtl_q_et_valid_exps = mtl_exps
    mtl_q_et_test_exps = [f"{tmp}0" for tmp in mtl_q_et_valid_exps]
    # valid for q
    exps_q_et_valid = stl_q_exps + mtl_q_et_valid_exps
    exps_q_et_test = [f"{tmp}0" for tmp in stl_q_exps] + mtl_q_et_test_exps

    # valid for et
    exps_et_q_valid = stl_et_exps + mtl_q_et_valid_exps
    exps_et_q_test = [f"{tmp}0" for tmp in stl_et_exps] + mtl_q_et_test_exps

    result_dir = os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "app",
        "multi_task",
        "results",
    )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    cases_exps_legends_together = [
        "STL",
        "1",
        "1/3",
        "1/8",
        "1/24",
    ]

    exps_test_q_et_results_file = os.path.join(
        result_dir,
        "exps_q_et_test_results.npy",
    )
    exps_test_et_q_results_file = os.path.join(
        result_dir,
        "exps_et_q_test_results.npy",
    )
    exps_valid_q_et_results_file = os.path.join(
        result_dir,
        "exps_q_et_valid_results.npy",
    )
    exps_valid_et_q_results_file = os.path.join(
        result_dir,
        "exps_et_q_valid_results.npy",
    )
    exps_test_pred_q_file = os.path.join(
        result_dir,
        "exps_q_test_pred.npy",
    )
    exps_test_obs_q_file = os.path.join(
        result_dir,
        "exps_q_test_obs.npy",
    )
    exps_test_et_pred_file = os.path.join(
        result_dir,
        "exps_et_test_pred.npy",
    )
    exps_test_et_obs_file = os.path.join(
        result_dir,
        "exps_et_test_obs.npy",
    )

    if (
        os.path.exists(exps_test_q_et_results_file)
        and os.path.exists(exps_test_et_q_results_file)
        and os.path.exists(exps_valid_q_et_results_file)
        and os.path.exists(exps_valid_et_q_results_file)
    ):
        exps_q_et_results = np.load(exps_test_q_et_results_file, allow_pickle=True)
        exps_et_q_results = np.load(exps_test_et_q_results_file, allow_pickle=True)
        q_et_valid_inds = np.load(exps_valid_q_et_results_file, allow_pickle=True)
        et_q_valid_inds = np.load(exps_valid_et_q_results_file, allow_pickle=True)
        preds_q_lst = np.load(exps_test_pred_q_file, allow_pickle=True)
        obss_q_lst = np.load(exps_test_obs_q_file, allow_pickle=True)
        preds_et_lst = np.load(exps_test_et_pred_file, allow_pickle=True)
        obss_et_lst = np.load(exps_test_et_obs_file, allow_pickle=True)
    else:
        et_q_valid_inds, et_q_best_index_valid_best4et = read_multi_single_exps_results(
            exps_et_q_valid, var_idx=1, single_is_flow=False, flow_idx_in_mtl=0
        )
        q_et_valid_inds, q_et_best_index_valid = read_multi_single_exps_results(
            exps_q_et_valid
        )

        # q when best4q
        (
            exps_q_et_results,
            _,
            preds_q_lst,
            obss_q_lst,
        ) = read_multi_single_exps_results(
            exps_q_et_test, q_et_best_index_valid, return_value=True
        )

        # et when best4q
        (
            exps_et_q_results,
            _,
            preds_et_lst,
            obss_et_lst,
        ) = read_multi_single_exps_results(
            exps_et_q_test,
            q_et_best_index_valid,
            var_idx=1,
            single_is_flow=False,
            flow_idx_in_mtl=0,
            return_value=True,
        )
        np.save(exps_valid_q_et_results_file, q_et_valid_inds, allow_pickle=True)
        np.save(exps_valid_et_q_results_file, et_q_valid_inds, allow_pickle=True)
        np.save(exps_test_q_et_results_file, exps_q_et_results, allow_pickle=True)
        np.save(exps_test_et_q_results_file, exps_et_q_results, allow_pickle=True)
        np.save(exps_test_pred_q_file, preds_q_lst, allow_pickle=True)
        np.save(exps_test_obs_q_file, obss_q_lst, allow_pickle=True)
        np.save(exps_test_et_pred_file, preds_et_lst, allow_pickle=True)
        np.save(exps_test_et_obs_file, obss_et_lst, allow_pickle=True)

    diff_q = exps_q_et_results[1] - exps_q_et_results[0]
    diff_q_sort = np.argsort(diff_q)
    both_positive_q = np.where((exps_q_et_results[1] > 0) & (exps_q_et_results[0] > 0))[
        0
    ]
    diff_q_sort_idx = [i for i in diff_q_sort if i in both_positive_q]
    gage_id_file = os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        "camels_us_mtl_2001_2021_flow_screen.csv",
    )
    gage_ids = pd.read_csv(gage_id_file)
    t_lst = hydro_utils.t_range_days(["2016-10-01", "2021-10-01"])

    # plot the 2 better mtl and 1 better stl
    plot_ts(
        np.tile(t_lst, (3, 1)).tolist(),
        [
            preds_q_lst[0, diff_q_sort_idx[-1], :],
            preds_q_lst[1, diff_q_sort_idx[-1], :],
            obss_q_lst[0, diff_q_sort_idx[-1], :],
        ],
        leg_lst=["径流单变量模型预测", "多变量模型径流预测", "径流观测"],
        alpha=0.5,
        xlabel="日期",
        ylabel="径流（m$^3$/s）",
    )
    plt.savefig(
        os.path.join(
            result_dir,
            "abasin_mtl_stl_flow_ts.png",
        ),
        dpi=600,
    )

    plot_ts(
        np.tile(t_lst, (3, 1)).tolist(),
        [
            preds_et_lst[0, diff_q_sort_idx[-1], :],
            preds_et_lst[1, diff_q_sort_idx[-1], :],
            obss_et_lst[0, diff_q_sort_idx[-1], :],
        ],
        leg_lst=["蒸散发单变量模型预测", "多变量模型蒸散发预测", "蒸散发观测"],
        alpha=0.5,
        xlabel="日期",
        ylabel="蒸散发（mm/day）",
    )
    plt.savefig(
        os.path.join(
            result_dir,
            "abasin_mtl_stl_et_ts.png",
        ),
        dpi=600,
    )

    # plot boxes of NSEs for valid Q
    plot_multi_single_comp_flow_boxes(
        q_et_valid_inds[:-2],
        cases_exps_legends_together=cases_exps_legends_together,
        save_path=os.path.join(
            result_dir,
            "mtl_valid_flow_boxes.png",
        ),
        rotation=45,
    )

    # plot boxes of NSEs for valid ET
    plot_multi_single_comp_flow_boxes(
        et_q_valid_inds[:-2],
        cases_exps_legends_together=cases_exps_legends_together,
        save_path=os.path.join(
            result_dir,
            "mtl_valid_et_boxes.png",
        ),
        rotation=45,
    )
    # plot scatter with a 1:1 line to compare single-task and multi-task models
    plot_scatter_with_11line(
        exps_q_et_results[0],
        exps_q_et_results[1],
        # xlabel="NSE single-task",
        # ylabel="NSE multi-task",
        xlabel="径流单变量学习模型预测NSE",
        ylabel="多变量学习模型径流预测NSE",
    )
    plt.savefig(
        os.path.join(
            result_dir,
            "mtl_stl_flow_scatter_plot_with_11line.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )

    plot_scatter_with_11line(
        exps_et_q_results[0],
        exps_et_q_results[1],
        # xlabel="NSE single-task",
        # ylabel="NSE multi-task",
        xlabel="蒸散发单变量学习模型预测NSE",
        ylabel="多变量学习模型蒸散发预测NSE",
    )
    plt.savefig(
        os.path.join(
            result_dir,
            "mtl_stl_et_scatter_plot_with_11line.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )

    # plot boxes of NSEs for test Q
    plot_multi_single_comp_flow_boxes(
        exps_q_et_results[:-2],
        cases_exps_legends_together=cases_exps_legends_together,
        save_path=os.path.join(
            result_dir,
            "mtl_test_flow_boxes.png",
        ),
        rotation=45,
    )

    # plot boxes of NSEs for test ET
    plot_multi_single_comp_flow_boxes(
        exps_et_q_results[:-2],
        cases_exps_legends_together=cases_exps_legends_together,
        save_path=os.path.join(
            result_dir,
            "mtl_test_et_boxes.png",
        ),
        rotation=45,
    )

    # plot all metrics for stl and mtl exps
    # for ET
    plot_multi_metrics_for_stl_mtl(
        [exps_et_q_test[0], exps_et_q_test[1]], result_dir, var_obj="et"
    )
    # for Q
    plot_multi_metrics_for_stl_mtl([exps_q_et_test[0], exps_q_et_test[1]], result_dir)

    # plot map
    plot_mtl_results_map(
        [exps_q_et_results[0], exps_q_et_results[1]],
        ["Q", "MTL-Q"],
        ["o", "x"],
        os.path.join(
            result_dir,
            "better_flow_stl_mtl_cases_map.png",
        ),
    )
    # plot map
    plot_mtl_results_map(
        [exps_et_q_results[0], exps_et_q_results[1]],
        ["ET", "MTL-ET"],
        ["o", "x"],
        os.path.join(
            result_dir,
            "better_et_stl_mtl_cases_map.png",
        ),
    )


if __name__ == "__main__":
    print("The exps which will be shown in the figures are:")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mtl_exps",
        dest="mtl_exps",
        help="the ID of the MTL train experiments, such as expmtl001, expmtl002",
        nargs="+",
        type=str,
        default=["expmtl002", "expmtl001", "expmtl003", "expmtl004", "expmtl005"],
    )
    parser.add_argument(
        "--stl_q_exps",
        dest="stl_q_exps",
        help="the ID of the STL(streamflow) train experiment, such as expstlq001",
        nargs="+",
        type=str,
        default=["expstlq001"],
    )
    parser.add_argument(
        "--stl_et_exps",
        dest="stl_et_exps",
        help="the ID of the STL(streamflow) train experiment, such as expstlet001",
        nargs="+",
        type=str,
        default=["expstlet001"],
    )

    args = parser.parse_args()
    print(f"Your command arguments:{str(args)}")
    plot_for_prediction_performance(args)
