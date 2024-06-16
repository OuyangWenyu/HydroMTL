"""
Author: Wenyu Ouyang
Date: 2024-04-29 08:46:00
LastEditTime: 2024-06-15 15:39:06
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

from tqdm import tqdm


# Get the current directory
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from scripts.mtl_results_utils import (
    read_multi_single_exps_results,
)
from scripts.streamflow_utils import get_json_file
from hydromtl.models.trainer import stat_result
from hydromtl.visual.plot_stat import plot_scatter_with_11line
from scripts.evaluate_ensemble import (
    get_ensemble_results,
    get_exps_of_diff_random_seed,
    load_from_cache,
    read_ensemble_metrics,
    save_to_cache,
)

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


def compare_et_pred():
    exps_lst = ["expstlet9010", "expstlet0010"]
    inds_df_lst = []
    for exp in exps_lst:
        cfg_dir_flow_other = os.path.join(definitions.RESULT_DIR, "camels", exp)
        cfg_flow_other = get_json_file(cfg_dir_flow_other)
        inds_df_, pred, obs = stat_result(
            cfg_flow_other["data_params"]["test_path"],
            cfg_flow_other["evaluate_params"]["test_epoch"],
            fill_nan=cfg_flow_other["evaluate_params"]["fill_nan"],
            return_value=True,
        )
        inds_df_lst.append(inds_df_)
    return inds_df_lst


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
    random_seed="1234",
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
    """
    if random_seed == "ensemble":
        plot_scatter_with_11line(
            exps_q_et_results[0][metric],
            exps_q_et_results[chosen_idx][metric],
            xlabel=f"STL_Q {metric}",
            ylabel=f"MTL_Q {metric}",
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
            np.array(exps_et_q_results[0][metric]),
            np.array(exps_et_q_results[chosen_idx][metric]),
            xlabel=f"STL_ET {metric}",
            ylabel=f"MTL_ET {metric}",
        )

        plt.savefig(
            os.path.join(
                figure_dir,
                f"mtl_stl_et_scatter_plot_with_11line_{figure_weight_ratios_names[chosen_idx]}_{metric}_{random_seed}.png",
            ),
            dpi=600,
            bbox_inches="tight",
        )
        return
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
    parser.add_argument("--metric", type=str, default="KGE", help="Performance metric")
    parser.add_argument(
        "--random_seed",
        type=str,
        default="ensemble",
        help="Random seed for the experiment or ensemble using all random seeds",
    )
    args = parser.parse_args()
    # compare_et_pred()
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
        if args.random_seed == "ensemble":

            (
                preds_q_test_ensemble,
                obss_q_test_ensemble,
                preds_et_test_ensemble,
                obss_et_test_ensemble,
                preds_q_valid_ensemble,
                obss_q_valid_ensemble,
                preds_et_valid_ensemble,
                obss_et_valid_ensemble,
                preds_q_train_ensemble,
                obss_q_train_ensemble,
                preds_et_train_ensemble,
                obss_et_train_ensemble,
            ) = get_ensemble_results()
            cache_files = {
                "inds_ensemble_q_valid": os.path.join(
                    definitions.RESULT_DIR, "cache", "inds_ensemble_q_valid.pkl"
                ),
                "inds_ensemble_q_test": os.path.join(
                    definitions.RESULT_DIR, "cache", "inds_ensemble_q_test.pkl"
                ),
                "inds_ensemble_et_valid": os.path.join(
                    definitions.RESULT_DIR, "cache", "inds_ensemble_et_valid.pkl"
                ),
                "inds_ensemble_et_test": os.path.join(
                    definitions.RESULT_DIR, "cache", "inds_ensemble_et_test.pkl"
                ),
            }
            inds_ensemble_q_test = load_from_cache(cache_files["inds_ensemble_q_test"])
            if inds_ensemble_q_test is None:
                inds_ensemble_q_test = read_ensemble_metrics(
                    preds_q_test_ensemble,
                    obss_q_test_ensemble,
                    cases_exps_legends_together,
                )
                save_to_cache(inds_ensemble_q_test, cache_files["inds_ensemble_q_test"])
            inds_ensemble_et_test = load_from_cache(
                cache_files["inds_ensemble_et_test"]
            )
            if inds_ensemble_et_test is None:
                inds_ensemble_et_test = read_ensemble_metrics(
                    preds_et_test_ensemble,
                    obss_et_test_ensemble,
                    cases_exps_legends_together,
                    var_idx=1,
                )
                save_to_cache(
                    inds_ensemble_et_test, cache_files["inds_ensemble_et_test"]
                )
                # 1 to 5 means 5 weight ratios
            for chosen_idx in tqdm(range(1, 6)):
                plot_scatter_with_11line_for_1metric1lossweightratio(
                    inds_ensemble_q_test,
                    inds_ensemble_et_test,
                    chosen_idx,
                    args.metric,
                    random_seed=args.random_seed,
                )
        else:
            (
                exps_q_et_valid,
                exps_et_q_valid,
                exps_q_et_test,
                exps_et_q_test,
                exps_q_et_train,
                exps_et_q_train,
            ) = get_exps_of_diff_random_seed(args.random_seed)
            exps_q_et_results, exps_et_q_results = one_metric_data_reader(
                exps_q_et_test,
                exps_et_q_test,
                args.metric,
                random_seed=args.random_seed,
            )
            print("Data reading complete.")
            # 1 to 5 means 5 weight ratios
            for chosen_idx in tqdm(range(1, 6)):
                plot_scatter_with_11line_for_1metric1lossweightratio(
                    exps_q_et_results,
                    exps_et_q_results,
                    chosen_idx,
                    args.metric,
                    random_seed=args.random_seed,
                )
