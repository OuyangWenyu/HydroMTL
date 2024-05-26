"""
Author: Wenyu Ouyang
Date: 2024-04-29 08:46:00
LastEditTime: 2024-05-26 11:23:01
LastEditors: Wenyu Ouyang
Description: According to reviewers' feedback, we add more results evaluation to the original notebook evaluate.ipynb.
    The performance metrics for evapotranspiration (ET) and streamflow (Q) under varying weights
    We create a comprehensive scatter diagram that displays the performance metrics for evapotranspiration (ET) and streamflow (Q) under varying weights.
    There are multiple figures to present the results. We plot three performance metrics (RMSE, NSE, and correlation) for ET and Q across different weights.
    For each weight, we calculate the RMSE, NSE, and correlation for ET and Q. We then plot the results in scatter diagrams.
    We have five varying weights in total: 2, 1, 1/3, 1/8, and 1/24; two variables and three performance metrics. Therefore, we have 5 * 2 * 3 = 30 scatter diagrams in total.
FilePath: \HydroMTL\scripts\evaluate_more.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import argparse


# Get the current directory
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from scripts.mtl_results_utils import (
    read_multi_single_exps_results,
)
from hydromtl.visual.plot_stat import plot_scatter_with_11line
from scripts.evaluate_ensemble import get_exps_of_diff_random_seed

result_cache_dir = os.path.join(
    definitions.RESULT_DIR,
    "cache",
)
figure_dir = os.path.join(
    definitions.RESULT_DIR,
    "figures",
    "evaluate_more",
)
if not os.path.exists(result_cache_dir):
    os.makedirs(result_cache_dir)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
cases_exps_legends_together = [
    "STL",
    "2",
    "1",
    "1/3",
    "1/8",
    "1/24",
]
figure_weight_ratios_names = [
    "STL",
    "2",
    "1",
    "one3rd",
    "one8th",
    "one24th",
]


def one_metric_data_reader(
    exps_q_et_test, exps_et_q_test, metric="NSE", random_seed=1234
):
    exps_test_q_et_results_file = os.path.join(
        result_cache_dir,
        f"exps_q_et_test_results_{metric}_{random_seed}.npy",
    )
    exps_test_et_q_results_file = os.path.join(
        result_cache_dir,
        f"exps_et_q_test_results_{metric}_{random_seed}.npy",
    )
    exps_test_pred_q_file = os.path.join(
        result_cache_dir,
        f"exps_q_test_pred_{metric}_{random_seed}.npy",
    )
    exps_test_obs_q_file = os.path.join(
        result_cache_dir,
        f"exps_q_test_obs_{metric}_{random_seed}.npy",
    )
    exps_test_et_pred_file = os.path.join(
        result_cache_dir,
        f"exps_et_test_pred_{metric}_{random_seed}.npy",
    )
    exps_test_et_obs_file = os.path.join(
        result_cache_dir,
        f"exps_et_test_obs_{metric}_{random_seed}.npy",
    )

    if os.path.exists(exps_test_q_et_results_file) and os.path.exists(
        exps_test_et_q_results_file
    ):
        exps_q_et_results = np.load(exps_test_q_et_results_file, allow_pickle=True)
        exps_et_q_results = np.load(exps_test_et_q_results_file, allow_pickle=True)
        preds_q_lst = np.load(exps_test_pred_q_file, allow_pickle=True)
        obss_q_lst = np.load(exps_test_obs_q_file, allow_pickle=True)
        preds_et_lst = np.load(exps_test_et_pred_file, allow_pickle=True)
        obss_et_lst = np.load(exps_test_et_obs_file, allow_pickle=True)
    else:
        # q when best4q
        (
            exps_q_et_results,
            _,
            preds_q_lst,
            obss_q_lst,
        ) = read_multi_single_exps_results(
            exps_q_et_test, return_value=True, metric=metric, ensemble=-1
        )

        # et when best4q
        exps_et_q_results, _, preds_et_lst, obss_et_lst = (
            read_multi_single_exps_results(
                exps_et_q_test, var_idx=1, return_value=True, metric=metric, ensemble=-1
            )
        )
        np.save(exps_test_q_et_results_file, exps_q_et_results, allow_pickle=True)
        np.save(exps_test_et_q_results_file, exps_et_q_results, allow_pickle=True)
        np.save(exps_test_pred_q_file, preds_q_lst, allow_pickle=True)
        np.save(exps_test_obs_q_file, obss_q_lst, allow_pickle=True)
        np.save(exps_test_et_pred_file, preds_et_lst, allow_pickle=True)
        np.save(exps_test_et_obs_file, obss_et_lst, allow_pickle=True)
    return exps_q_et_results, exps_et_q_results


def plot_scatter_with_11line_for_1metric1lossweightratio(
    exps_q_et_results,
    exps_et_q_results,
    chosen_idx,
    metric="NSE",
    maxdiff_label=False,
    random_seed=1234,
):
    """_summary_

    Parameters
    ----------
    exps_q_et_results : _type_
        one metric results for Q
    exps_et_q_results : _type_
        one metric results for ET
    chosen_idx : int
        each number means a loss weight ratio;
        0 means STL, 1 means 2, 2 means 1, 3 means 1/3, 4 means 1/8, 5 means 1/24
    metric
        the metric to be evaluated
    maxdiff_label
        whether to label the point with the most significant difference
    """
    chosen_mtl4q_test_result = exps_q_et_results[chosen_idx]
    chosen_mtl4et_test_result = exps_et_q_results[chosen_idx]

    # plot scatter with a 1:1 line to compare single-task and multi-task models
    plot_scatter_with_11line(
        exps_q_et_results[0],
        chosen_mtl4q_test_result,
        # xlabel="NSE single-task",
        # ylabel="NSE multi-task",
        xlabel=f"STL_Q {metric}",
        ylabel=f"MTL_Q {metric}",
    )
    if maxdiff_label:
        mark_color = "darkred"

        # Extract the first and second 1-D arrays
        x = exps_q_et_results[0]
        y = chosen_mtl4q_test_result
        # Filter the data to only include points where both x and y are in the range [0, 1]
        mask = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        filtered_x = x[mask]
        filtered_y = y[mask]

        # Calculate the difference for the filtered data
        filtered_diff = np.abs(filtered_x - filtered_y)

        # Find the index of the point with the most significant difference in the filtered data
        filtered_max_diff_index = np.argmax(filtered_diff)

        # Highlight the point with the most significant difference with a red circle
        plt.gca().add_artist(
            plt.Circle(
                (
                    filtered_x[filtered_max_diff_index],
                    filtered_y[filtered_max_diff_index],
                ),
                0.02,
                fill=False,
                color=mark_color,
                linewidth=2,
            )
        )
        # Label the plot
        plt.text(
            filtered_x[filtered_max_diff_index],
            filtered_y[filtered_max_diff_index],
            " Max diff",
            verticalalignment="bottom",
            horizontalalignment="left",
            color=mark_color,
            fontsize=18,
        )

    plt.savefig(
        os.path.join(
            figure_dir,
            f"mtl_stl_flow_scatter_plot_with_11line_{figure_weight_ratios_names[chosen_idx]}_{metric}_{random_seed}.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )

    plot_scatter_with_11line(
        exps_et_q_results[0],
        chosen_mtl4et_test_result,
        # xlabel="NSE single-task",
        # ylabel="NSE multi-task",
        xlabel=f"STL_ET {metric}",
        ylabel=f"MTL_ET {metric}",
    )
    if maxdiff_label:
        # Extract the first and second 1-D arrays from the second 2-D array
        x2 = exps_et_q_results[0]
        y2 = chosen_mtl4et_test_result

        # Find the index of the point with the max difference in the first plot in the second plot
        index_in_second_plot = np.where(mask)[0][filtered_max_diff_index]
        # Highlight the point with the same index as the point with the max difference in the first plot
        plt.gca().add_artist(
            plt.Circle(
                (x2[index_in_second_plot], y2[index_in_second_plot]),
                0.01,
                fill=False,
                color=mark_color,
                linewidth=2,
            )
        )

        # Label the plot
        plt.text(
            x2[index_in_second_plot],
            y2[index_in_second_plot],
            " Max diff \n in fig(a)",
            verticalalignment="top",
            horizontalalignment="left",
            color=mark_color,
            fontsize=18,
        )

    plt.savefig(
        os.path.join(
            figure_dir,
            f"mtl_stl_et_scatter_plot_with_11line_{figure_weight_ratios_names[chosen_idx]}_{metric}_{random_seed}.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="NSE", help="Performance metric")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=12345,
        help="Random seed for the experiment",
    )
    args = parser.parse_args()

    available_metrics = [
        "Bias",
        "RMSE",
        "ubRMSE",
        "Corr",
        "R2",
        "NSE",
        "KGE",
        "FHV",
        "FLV",
    ]
    if args.metric not in available_metrics:
        print("Invalid metric. Please choose from the available metrics:")
        print(available_metrics)
    else:
        print("Reading data...")
        (
            exps_q_et_valid,
            exps_et_q_valid,
            exps_q_et_test,
            exps_et_q_test,
            exps_q_et_train,
            exps_et_q_train,
        ) = get_exps_of_diff_random_seed(args.random_seed)
        exps_q_et_results, exps_et_q_results = one_metric_data_reader(
            exps_q_et_test, exps_et_q_test, args.metric, random_seed=args.random_seed
        )
        print("Data reading complete.")
        # 1 to 5 means 5 weight ratios
        # for chosen_idx in tqdm(range(1, 6)):
        #     plot_scatter_with_11line_for_1metric1lossweightratio(
        #         exps_q_et_results,
        #         exps_et_q_results,
        #         chosen_idx,
        #         args.metric,
        #         random_seed=args.random_seed,
        #     )
