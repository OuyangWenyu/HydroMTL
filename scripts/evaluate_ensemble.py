"""
Author: Wenyu Ouyang
Date: 2024-05-09 16:07:19
LastEditTime: 2024-05-26 15:19:27
LastEditors: Wenyu Ouyang
Description: Same content with evaluate.ipynb but in .py format
FilePath: \HydroMTL\scripts\evaluate_ensemble.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from matplotlib import pyplot as plt
import numpy as np
import os
import sys


# Get the project directory of the py file
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from scripts.mtl_results_utils import (
    read_multi_single_exps_results,
)
from scripts.evaluate import (
    plot_multi_metrics,
    plot_scatter,
    plot_test_boxes,
    plot_valid_boxes,
)


# set font
plt.rcParams["font.family"] = "Times New Roman"


# ---------------------------------- Read Results ---------------------------------------
def get_exps_of_diff_random_seed(random_seed=1234):
    # MTL exps with different λ: 2, 1, 1/3, 1/8, 1/24
    if random_seed == 1234:
        mtl_q_et_valid_exps = [
            "expmtl002",
            "expmtl001",
            "expmtl003",
            "expmtl004",
            "expmtl005",
        ]
        mtl_q_et_test_exps = [f"{tmp}0" for tmp in mtl_q_et_valid_exps]
        # index 0 is STL-Q, index 1-5 are MTL-Q
        exps_q_et_valid = ["expstlq001"] + mtl_q_et_valid_exps
        exps_q_et_test = ["expstlq0010"] + mtl_q_et_test_exps

        # index 0 is STL-ET, index 1-5 are MTL-Q
        exps_et_q_valid = ["expstlet001"] + mtl_q_et_valid_exps
        exps_et_q_test = ["expstlet0010"] + mtl_q_et_test_exps

        # evaluate train results
        mtl_q_et_train_exps = [
            "expmtltrain002",
            "expmtltrain001",
            "expmtltrain003",
            "expmtltrain004",
            "expmtltrain005",
        ]
        exps_q_et_train = ["expstlqtrain001"] + mtl_q_et_train_exps
        exps_et_q_train = ["expstlettrain001"] + mtl_q_et_train_exps
    elif random_seed == 12345:
        mtl_q_et_valid_exps = [
            "expmtl202",
            "expmtl201",
            "expmtl203",
            "expmtl204",
            "expmtl205",
        ]
        mtl_q_et_test_exps = [f"{tmp}0" for tmp in mtl_q_et_valid_exps]
        # index 0 is STL-Q, index 1-5 are MTL-Q
        exps_q_et_valid = ["expstlq202"] + mtl_q_et_valid_exps
        exps_q_et_test = ["expstlq2020"] + mtl_q_et_test_exps

        # index 0 is STL-ET, index 1-5 are MTL-Q
        exps_et_q_valid = ["expstlet003"] + mtl_q_et_valid_exps
        exps_et_q_test = ["expstlet0030"] + mtl_q_et_test_exps

        # evaluate train results
        mtl_q_et_train_exps = [
            "expmtltrain202",
            "expmtltrain201",
            "expmtltrain203",
            "expmtltrain204",
            "expmtltrain205",
        ]
        exps_q_et_train = ["expstlqtrain202"] + mtl_q_et_train_exps
        exps_et_q_train = ["expstlettrain003"] + mtl_q_et_train_exps
    elif random_seed == 123:
        mtl_q_et_valid_exps = [
            "expmtl302",
            "expmtl301",
            "expmtl303",
            "expmtl304",
            "expmtl305",
        ]
        mtl_q_et_test_exps = [f"{tmp}0" for tmp in mtl_q_et_valid_exps]
        # index 0 is STL-Q, index 1-5 are MTL-Q
        exps_q_et_valid = ["expstlq203"] + mtl_q_et_valid_exps
        exps_q_et_test = ["expstlq2030"] + mtl_q_et_test_exps

        # index 0 is STL-ET, index 1-5 are MTL-Q
        exps_et_q_valid = ["expstlet002"] + mtl_q_et_valid_exps
        exps_et_q_test = ["expstlet0020"] + mtl_q_et_test_exps

        # evaluate train results
        mtl_q_et_train_exps = [
            "expmtltrain302",
            "expmtltrain301",
            "expmtltrain303",
            "expmtltrain304",
            "expmtltrain305",
        ]
        exps_q_et_train = ["expstlqtrain203"] + mtl_q_et_train_exps
        exps_et_q_train = ["expstlettrain002"] + mtl_q_et_train_exps
    elif random_seed == 111:
        mtl_q_et_valid_exps = [
            "expmtl402",
            "expmtl401",
            "expmtl403",
            "expmtl404",
            "expmtl405",
        ]
        mtl_q_et_test_exps = [f"{tmp}0" for tmp in mtl_q_et_valid_exps]
        # index 0 is STL-Q, index 1-5 are MTL-Q
        exps_q_et_valid = ["expstlq204"] + mtl_q_et_valid_exps
        exps_q_et_test = ["expstlq2040"] + mtl_q_et_test_exps

        # index 0 is STL-ET, index 1-5 are MTL-Q
        exps_et_q_valid = ["expstlet004"] + mtl_q_et_valid_exps
        exps_et_q_test = ["expstlet0040"] + mtl_q_et_test_exps

        # evaluate train results
        mtl_q_et_train_exps = [
            "expmtltrain402",
            "expmtltrain401",
            "expmtltrain403",
            "expmtltrain404",
            "expmtltrain405",
        ]
        exps_q_et_train = ["expstlqtrain204"] + mtl_q_et_train_exps
        exps_et_q_train = ["expstlettrain004"] + mtl_q_et_train_exps
    elif random_seed == 1111:
        mtl_q_et_valid_exps = [
            "expmtl502",
            "expmtl501",
            "expmtl503",
            "expmtl504",
            "expmtl505",
        ]
        mtl_q_et_test_exps = [f"{tmp}0" for tmp in mtl_q_et_valid_exps]
        # index 0 is STL-Q, index 1-5 are MTL-Q
        exps_q_et_valid = ["expstlq205"] + mtl_q_et_valid_exps
        exps_q_et_test = ["expstlq2050"] + mtl_q_et_test_exps

        # index 0 is STL-ET, index 1-5 are MTL-Q
        exps_et_q_valid = ["expstlet005"] + mtl_q_et_valid_exps
        exps_et_q_test = ["expstlet0050"] + mtl_q_et_test_exps

        # evaluate train results
        mtl_q_et_train_exps = [
            "expmtltrain502",
            "expmtltrain501",
            "expmtltrain503",
            "expmtltrain504",
            "expmtltrain505",
        ]
        exps_q_et_train = ["expstlqtrain205"] + mtl_q_et_train_exps
        exps_et_q_train = ["expstlettrain005"] + mtl_q_et_train_exps
    return (
        exps_q_et_valid,
        exps_et_q_valid,
        exps_q_et_test,
        exps_et_q_test,
        exps_q_et_train,
        exps_et_q_train,
    )


def get_results(
    exps_q_et_valid, exps_et_q_valid, exps_q_et_test, exps_et_q_test, random_seed=1234
):
    result_cache_dir = os.path.join(
        definitions.RESULT_DIR,
        "cache",
    )
    if not os.path.exists(result_cache_dir):
        os.makedirs(result_cache_dir)

    exps_test_q_et_results_file = os.path.join(
        result_cache_dir,
        f"exps_q_et_test_results_{random_seed}.npy",
    )
    exps_test_et_q_results_file = os.path.join(
        result_cache_dir,
        f"exps_et_q_test_results_{random_seed}.npy",
    )
    exps_valid_q_et_results_file = os.path.join(
        result_cache_dir,
        f"exps_q_et_valid_results_{random_seed}.npy",
    )
    exps_valid_et_q_results_file = os.path.join(
        result_cache_dir,
        f"exps_et_q_valid_results_{random_seed}.npy",
    )

    exps_test_pred_q_file = os.path.join(
        result_cache_dir,
        f"exps_q_test_pred_{random_seed}.npy",
    )
    exps_test_obs_q_file = os.path.join(
        result_cache_dir,
        f"exps_q_test_obs_{random_seed}.npy",
    )
    exps_test_et_pred_file = os.path.join(
        result_cache_dir,
        f"exps_et_test_pred_{random_seed}.npy",
    )
    exps_test_et_obs_file = os.path.join(
        result_cache_dir,
        f"exps_et_test_obs_{random_seed}.npy",
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
            exps_et_q_valid,
            var_idx=1,
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
        exps_et_q_results, _, preds_et_lst, obss_et_lst = (
            read_multi_single_exps_results(
                exps_et_q_test,
                q_et_best_index_valid,
                var_idx=1,
                return_value=True,
            )
        )
        np.save(exps_valid_q_et_results_file, q_et_valid_inds, allow_pickle=True)
        np.save(exps_valid_et_q_results_file, et_q_valid_inds, allow_pickle=True)
        np.save(exps_test_q_et_results_file, exps_q_et_results, allow_pickle=True)
        np.save(exps_test_et_q_results_file, exps_et_q_results, allow_pickle=True)
        np.save(exps_test_pred_q_file, preds_q_lst, allow_pickle=True)
        np.save(exps_test_obs_q_file, obss_q_lst, allow_pickle=True)
        np.save(exps_test_et_pred_file, preds_et_lst, allow_pickle=True)
        np.save(exps_test_et_obs_file, obss_et_lst, allow_pickle=True)
    return (
        exps_q_et_results,
        exps_et_q_results,
        q_et_valid_inds,
        et_q_valid_inds,
        preds_q_lst,
        obss_q_lst,
        preds_et_lst,
        obss_et_lst,
    )


if __name__ == "__main__":
    random_seed = 1111
    (
        exps_q_et_valid,
        exps_et_q_valid,
        exps_q_et_test,
        exps_et_q_test,
        exps_q_et_train,
        exps_et_q_train,
    ) = get_exps_of_diff_random_seed(random_seed)
    (
        exps_q_et_results,
        exps_et_q_results,
        q_et_valid_inds,
        et_q_valid_inds,
        preds_q_lst,
        obss_q_lst,
        preds_et_lst,
        obss_et_lst,
    ) = get_results(
        exps_q_et_valid,
        exps_et_q_valid,
        exps_q_et_test,
        exps_et_q_test,
        random_seed=random_seed,
    )
    cases_exps_legends_together = [
        "STL",
        "2",
        "1",
        "1/3",
        "1/8",
        "1/24",
    ]
    figure_dir = os.path.join(
        definitions.RESULT_DIR,
        "figures",
    )
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    # ----------------------  Plot for valid period  ---------------------
    # plot boxes of NSEs for valid Q

    plot_valid_boxes(
        figure_dir,
        cases_exps_legends_together,
        q_et_valid_inds,
        et_q_valid_inds,
        random_seed=random_seed,
    )

    # -- Plot for testing period. 1/3 seems best as it didn't change make any variable's prediction worse.--
    chosen_idx = 3
    chosen_mtl4q_test_result = exps_q_et_results[chosen_idx]
    chosen_mtl4et_test_result = exps_et_q_results[chosen_idx]

    # plot boxes of NSEs for test Q
    # plot_test_boxes(
    #     figure_dir,
    #     cases_exps_legends_together,
    #     exps_q_et_results,
    #     exps_et_q_results,
    #     random_seed=random_seed,
    # )

    # --------------- Plot all metrics for testing period ---------------------------
    # plot all metrics for stl and mtl exps
    # for ET
    plot_multi_metrics(
        exps_q_et_test, exps_et_q_test, figure_dir, chosen_idx, random_seed=random_seed
    )

    # plot scatter with a 1:1 line to compare single-task and multi-task models
    plot_scatter(
        figure_dir,
        exps_q_et_results,
        exps_et_q_results,
        chosen_mtl4q_test_result,
        chosen_mtl4et_test_result,
        random_seed=random_seed,
    )

    # ---- Plot time-series for some specific basins ------
    # plot_ts_figures(
    #     figure_dir,
    #     exps_q_et_results,
    #     preds_q_lst,
    #     obss_q_lst,
    #     preds_et_lst,
    #     obss_et_lst,
    #     preds_q_train_lst,
    #     obss_q_train_lst,
    #     preds_et_train_lst,
    #     obss_et_train_lst,
    #     chosen_idx,
    #     chosen_mtl4q_test_result,
    # )

    # ----------------------  Plot maps -------------------------
    # plot map
    # plot_map_figures(
    #     figure_dir,
    #     exps_q_et_results,
    #     exps_et_q_results,
    #     chosen_mtl4q_test_result,
    #     chosen_mtl4et_test_result,
    # )
