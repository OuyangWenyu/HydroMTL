"""
Author: Wenyu Ouyang
Date: 2024-05-09 16:07:19
LastEditTime: 2024-06-17 09:44:33
LastEditors: Wenyu Ouyang
Description: Same content with evaluate.ipynb but in .py format
FilePath: \HydroMTL\scripts\evaluate_ensemble.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import pickle


# Get the project directory of the py file
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from hydromtl.visual.plot_stat import plot_boxes_matplotlib
from hydromtl.utils.hydro_stat import stat_error
from scripts.mtl_results_utils import (
    plot_multi_metrics_for_stl_mtl,
    plot_multi_single_comp_flow_boxes,
    read_multi_single_exps_results,
)
from scripts.evaluate import (
    plot_map_figures,
    plot_scatter,
    plot_ts_figures,
)
from scripts.general_vars import random_seeds


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
    exps_q_et_train,
    exps_et_q_train,
    exps_q_et_valid,
    exps_et_q_valid,
    exps_q_et_test,
    exps_et_q_test,
    random_seed=1234,
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
    exps_train_q_et_results_file = os.path.join(
        result_cache_dir,
        f"exps_q_et_train_results_{random_seed}.npy",
    )
    exps_train_et_q_results_file = os.path.join(
        result_cache_dir,
        f"exps_et_q_train_results_{random_seed}.npy",
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
    exps_valid_pred_q_file = os.path.join(
        result_cache_dir,
        f"exps_q_valid_pred_{random_seed}.npy",
    )
    exps_valid_obs_q_file = os.path.join(
        result_cache_dir,
        f"exps_q_valid_obs_{random_seed}.npy",
    )
    exps_valid_et_pred_file = os.path.join(
        result_cache_dir,
        f"exps_et_valid_pred_{random_seed}.npy",
    )
    exps_valid_et_obs_file = os.path.join(
        result_cache_dir,
        f"exps_et_valid_obs_{random_seed}.npy",
    )
    exps_train_q_pred_file = os.path.join(
        result_cache_dir,
        f"exps_q_train_pred_{random_seed}.npy",
    )
    exps_train_q_obs_file = os.path.join(
        result_cache_dir,
        f"exps_q_train_obs_{random_seed}.npy",
    )
    exps_train_et_pred_file = os.path.join(
        result_cache_dir,
        f"exps_et_train_pred_{random_seed}.npy",
    )
    exps_train_et_obs_file = os.path.join(
        result_cache_dir,
        f"exps_et_train_obs_{random_seed}.npy",
    )
    if (
        # check if the cache files exist, we didn't check all files, so make sure run the code successfully once
        os.path.exists(exps_test_q_et_results_file)
        and os.path.exists(exps_test_et_q_results_file)
        and os.path.exists(exps_valid_q_et_results_file)
        and os.path.exists(exps_valid_et_q_results_file)
        and os.path.exists(exps_valid_pred_q_file)
        and os.path.exists(exps_valid_obs_q_file)
        and os.path.exists(exps_valid_et_pred_file)
        and os.path.exists(exps_valid_et_obs_file)
        and os.path.exists(exps_train_q_et_results_file)
        and os.path.exists(exps_train_et_q_results_file)
    ):
        q_et_test_inds = np.load(exps_test_q_et_results_file, allow_pickle=True)
        et_q_test_inds = np.load(exps_test_et_q_results_file, allow_pickle=True)
        q_et_valid_inds = np.load(exps_valid_q_et_results_file, allow_pickle=True)
        et_q_valid_inds = np.load(exps_valid_et_q_results_file, allow_pickle=True)
        q_et_train_inds = np.load(exps_train_q_et_results_file, allow_pickle=True)
        et_q_train_inds = np.load(exps_train_et_q_results_file, allow_pickle=True)
        preds_q_lst_test = np.load(exps_test_pred_q_file, allow_pickle=True)
        obss_q_lst_test = np.load(exps_test_obs_q_file, allow_pickle=True)
        preds_et_lst_test = np.load(exps_test_et_pred_file, allow_pickle=True)
        obss_et_lst_test = np.load(exps_test_et_obs_file, allow_pickle=True)
        preds_q_lst_valid = np.load(exps_valid_pred_q_file, allow_pickle=True)
        obss_q_lst_valid = np.load(exps_valid_obs_q_file, allow_pickle=True)
        preds_et_lst_valid = np.load(exps_valid_et_pred_file, allow_pickle=True)
        obss_et_lst_valid = np.load(exps_valid_et_obs_file, allow_pickle=True)
        preds_q_lst_train = np.load(exps_train_q_pred_file, allow_pickle=True)
        obss_q_lst_train = np.load(exps_train_q_obs_file, allow_pickle=True)
        preds_et_lst_train = np.load(exps_train_et_pred_file, allow_pickle=True)
        obss_et_lst_train = np.load(exps_train_et_obs_file, allow_pickle=True)
    else:
        (
            q_et_train_inds,
            _,
            preds_q_lst_train,
            obss_q_lst_train,
        ) = read_multi_single_exps_results(
            exps_q_et_train,
            return_value=True,
        )
        (et_q_train_inds, _, preds_et_lst_train, obss_et_lst_train) = (
            read_multi_single_exps_results(
                exps_et_q_train,
                var_idx=1,
                return_value=True,
            )
        )
        (
            et_q_valid_inds,
            et_q_best_index_valid_best4et,
            preds_et_lst_valid,
            obss_et_lst_valid,
        ) = read_multi_single_exps_results(
            exps_et_q_valid,
            var_idx=1,
            return_value=True,
        )
        q_et_valid_inds, q_et_best_index_valid, preds_q_lst_valid, obss_q_lst_valid = (
            read_multi_single_exps_results(
                exps_q_et_valid,
                return_value=True,
            )
        )

        # q when best4q
        (
            q_et_test_inds,
            _,
            preds_q_lst_test,
            obss_q_lst_test,
        ) = read_multi_single_exps_results(
            exps_q_et_test, q_et_best_index_valid, return_value=True
        )

        # et when best4q
        et_q_test_inds, _, preds_et_lst_test, obss_et_lst_test = (
            read_multi_single_exps_results(
                exps_et_q_test,
                q_et_best_index_valid,
                var_idx=1,
                return_value=True,
            )
        )
        np.save(exps_valid_q_et_results_file, q_et_valid_inds, allow_pickle=True)
        np.save(exps_valid_et_q_results_file, et_q_valid_inds, allow_pickle=True)
        np.save(exps_test_q_et_results_file, q_et_test_inds, allow_pickle=True)
        np.save(exps_test_et_q_results_file, et_q_test_inds, allow_pickle=True)
        np.save(exps_train_q_et_results_file, q_et_train_inds, allow_pickle=True)
        np.save(exps_train_et_q_results_file, et_q_train_inds, allow_pickle=True)
        np.save(exps_test_pred_q_file, preds_q_lst_test, allow_pickle=True)
        np.save(exps_test_obs_q_file, obss_q_lst_test, allow_pickle=True)
        np.save(exps_test_et_pred_file, preds_et_lst_test, allow_pickle=True)
        np.save(exps_test_et_obs_file, obss_et_lst_test, allow_pickle=True)
        np.save(exps_valid_pred_q_file, preds_q_lst_valid, allow_pickle=True)
        np.save(exps_valid_obs_q_file, obss_q_lst_valid, allow_pickle=True)
        np.save(exps_valid_et_pred_file, preds_et_lst_valid, allow_pickle=True)
        np.save(exps_valid_et_obs_file, obss_et_lst_valid, allow_pickle=True)
        np.save(exps_train_q_pred_file, preds_q_lst_train, allow_pickle=True)
        np.save(exps_train_q_obs_file, obss_q_lst_train, allow_pickle=True)
        np.save(exps_train_et_pred_file, preds_et_lst_train, allow_pickle=True)
        np.save(exps_train_et_obs_file, obss_et_lst_train, allow_pickle=True)
    return (
        q_et_test_inds,
        et_q_test_inds,
        q_et_valid_inds,
        et_q_valid_inds,
        q_et_train_inds,
        et_q_train_inds,
        preds_q_lst_test,
        obss_q_lst_test,
        preds_et_lst_test,
        obss_et_lst_test,
        preds_q_lst_valid,
        obss_q_lst_valid,
        preds_et_lst_valid,
        obss_et_lst_valid,
        preds_q_lst_train,
        obss_q_lst_train,
        preds_et_lst_train,
        obss_et_lst_train,
    )


def get_ensemble_results():
    cases_exps_legends_together = [
        "STL",
        "2",
        "1",
        "1/3",
        "1/8",
        "1/24",
    ]
    preds_q_test_ensemble_lst = [[] for _ in range(len(cases_exps_legends_together))]
    preds_et_test_ensemble_lst = [[] for _ in range(len(cases_exps_legends_together))]
    preds_q_valid_ensemble_lst = [[] for _ in range(len(cases_exps_legends_together))]
    preds_et_valid_ensemble_lst = [[] for _ in range(len(cases_exps_legends_together))]
    preds_q_train_ensemble_lst = [[] for _ in range(len(cases_exps_legends_together))]
    preds_et_train_ensemble_lst = [[] for _ in range(len(cases_exps_legends_together))]
    for random_seed in random_seeds:
        (
            exps_q_et_valid,
            exps_et_q_valid,
            exps_q_et_test,
            exps_et_q_test,
            exps_q_et_train,
            exps_et_q_train,
        ) = get_exps_of_diff_random_seed(random_seed)
        (
            q_et_test_inds,
            et_q_test_inds,
            q_et_valid_inds,
            et_q_valid_inds,
            q_et_train_inds,
            et_q_train_inds,
            preds_q_lst_test,
            obss_q_lst_test,
            preds_et_lst_test,
            obss_et_lst_test,
            preds_q_lst_valid,
            obss_q_lst_valid,
            preds_et_lst_valid,
            obss_et_lst_valid,
            preds_q_lst_train,
            obss_q_lst_train,
            preds_et_lst_train,
            obss_et_lst_train,
        ) = get_results(
            exps_q_et_train,
            exps_et_q_train,
            exps_q_et_valid,
            exps_et_q_valid,
            exps_q_et_test,
            exps_et_q_test,
            random_seed=random_seed,
        )
        for case_idx in range(len(cases_exps_legends_together)):
            preds_q_test_ensemble_lst[case_idx].append(preds_q_lst_test[case_idx])
            preds_et_test_ensemble_lst[case_idx].append(preds_et_lst_test[case_idx])
            preds_q_valid_ensemble_lst[case_idx].append(preds_q_lst_valid[case_idx])
            preds_et_valid_ensemble_lst[case_idx].append(preds_et_lst_valid[case_idx])
            preds_q_train_ensemble_lst[case_idx].append(preds_q_lst_train[case_idx])
            preds_et_train_ensemble_lst[case_idx].append(preds_et_lst_train[case_idx])
    preds_et_test_ensemble = []
    preds_q_test_ensemble = []
    obss_et_test_ensemble = obss_et_lst_test
    obss_q_test_ensemble = obss_q_lst_test
    preds_et_valid_ensemble = []
    preds_q_valid_ensemble = []
    obss_et_valid_ensemble = obss_et_lst_valid
    obss_q_valid_ensemble = obss_q_lst_valid
    preds_et_train_ensemble = []
    preds_q_train_ensemble = []
    obss_et_train_ensemble = obss_et_lst_train
    obss_q_train_ensemble = obss_q_lst_train
    for i in range(len(cases_exps_legends_together)):
        ensemble_preds_q_test = np.mean(np.array(preds_q_test_ensemble_lst[i]), axis=0)
        ensemble_preds_et_test = np.mean(
            np.array(preds_et_test_ensemble_lst[i]), axis=0
        )
        preds_et_test_ensemble.append(ensemble_preds_et_test)
        preds_q_test_ensemble.append(ensemble_preds_q_test)
        ensemble_preds_q_valid = np.mean(
            np.array(preds_q_valid_ensemble_lst[i]), axis=0
        )
        ensemble_preds_et_valid = np.mean(
            np.array(preds_et_valid_ensemble_lst[i]), axis=0
        )
        preds_et_valid_ensemble.append(ensemble_preds_et_valid)
        preds_q_valid_ensemble.append(ensemble_preds_q_valid)
        ensemble_preds_q_train = np.mean(
            np.array(preds_q_train_ensemble_lst[i]), axis=0
        )
        ensemble_preds_et_train = np.mean(
            np.array(preds_et_train_ensemble_lst[i]), axis=0
        )
        preds_et_train_ensemble.append(ensemble_preds_et_train)
        preds_q_train_ensemble.append(ensemble_preds_q_train)
    return (
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
    )


def read_ensemble_metrics(
    pred_ensemble,
    obs_ensemble,
    cases_exps_legends_together,
    var_idx=0,
):
    fill_nan = ["no", "mean"]
    inds_all_lst = []
    for i in range(len(cases_exps_legends_together)):
        inds_ = stat_error(
            obs_ensemble[0],
            pred_ensemble[i],
            fill_nan=fill_nan[var_idx],
        )
        inds_all_lst.append(inds_)
    return inds_all_lst


def extract_and_aggregate_metrics(data_list, metric_key):
    """
    Extract lists of a specific metric from a list of dictionaries and aggregate them into a list of lists.

    Args:
        data_list (list of dict): A list where each element is a dictionary containing metric data.
        metric_key (str): The key for the metric to extract and aggregate.

    Returns:
        list: A list of lists, each containing the values for the specified metric from each dictionary.
    """
    aggregated_list = []
    for data in data_list:
        if metric_key in data:  # Ensure the key exists in the dictionary
            aggregated_list.append(
                data[metric_key]
            )  # Append the list from the current dictionary
    return aggregated_list


def extract_metrics_from_dict(data_list, dict_indices, metric_keys):
    """
    Extract specified metrics from multiple dictionaries at given indices in a list of dictionaries,
    and aggregate these metrics into a list of lists, each containing arrays for each metric.

    Args:
        data_list (list of dict): List of dictionaries containing metric data.
        dict_indices (list of int): Indices of the dictionaries to extract data from.
        metric_keys (list of str): List of keys for the metrics to extract.

    Returns:
        list: A list of lists, where each inner list contains NumPy arrays for that metric across the specified dictionaries.
    """
    # Initialize a list of lists for results, one sub-list for each metric
    results = [[] for _ in metric_keys]

    # Iterate over each specified index
    for index in dict_indices:
        if index >= len(data_list):
            raise IndexError(
                f"The specified index {index} is out of the range of the data list."
            )
        current_dict = data_list[index]

        # Iterate over each metric key and append data to the corresponding sub-list in results
        for idx, key in enumerate(metric_keys):
            if key in current_dict:
                # Convert list to numpy array before appending
                results[idx].append(np.array(current_dict[key]))
            else:
                # Append None or an empty numpy array if the key is not found
                results[idx].append(np.array([]))  # Adjust this as needed

    return results


def save_to_cache(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_from_cache(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return None


if __name__ == "__main__":
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
    cases_exps_legends_together = [
        "STL",
        "2",
        "1",
        "1/3",
        "1/8",
        "1/24",
    ]
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

    inds_ensemble_q_valid = load_from_cache(cache_files["inds_ensemble_q_valid"])
    if inds_ensemble_q_valid is None:
        inds_ensemble_q_valid = read_ensemble_metrics(
            preds_q_valid_ensemble,
            obss_q_valid_ensemble,
            cases_exps_legends_together,
        )
        save_to_cache(inds_ensemble_q_valid, cache_files["inds_ensemble_q_valid"])

    inds_ensemble_q_test = load_from_cache(cache_files["inds_ensemble_q_test"])
    if inds_ensemble_q_test is None:
        inds_ensemble_q_test = read_ensemble_metrics(
            preds_q_test_ensemble,
            obss_q_test_ensemble,
            cases_exps_legends_together,
        )
        save_to_cache(inds_ensemble_q_test, cache_files["inds_ensemble_q_test"])

    inds_ensemble_et_valid = load_from_cache(cache_files["inds_ensemble_et_valid"])
    if inds_ensemble_et_valid is None:
        inds_ensemble_et_valid = read_ensemble_metrics(
            preds_et_valid_ensemble,
            obss_et_valid_ensemble,
            cases_exps_legends_together,
            var_idx=1,
        )
        save_to_cache(inds_ensemble_et_valid, cache_files["inds_ensemble_et_valid"])

    inds_ensemble_et_test = load_from_cache(cache_files["inds_ensemble_et_test"])
    if inds_ensemble_et_test is None:
        inds_ensemble_et_test = read_ensemble_metrics(
            preds_et_test_ensemble,
            obss_et_test_ensemble,
            cases_exps_legends_together,
            var_idx=1,
        )
        save_to_cache(inds_ensemble_et_test, cache_files["inds_ensemble_et_test"])

    figure_dir = os.path.join(
        definitions.RESULT_DIR,
        "figures",
    )
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    # ----------------------  Plot for valid period  ---------------------
    # plot boxes of NSEs for valid Q
    valid_ensemble_q_nse = extract_and_aggregate_metrics(inds_ensemble_q_valid, "NSE")
    plot_multi_single_comp_flow_boxes(
        valid_ensemble_q_nse,
        cases_exps_legends_together=cases_exps_legends_together,
        save_path=os.path.join(
            figure_dir,
            "mtl_valid_flow_boxes_ensemble.png",
        ),
        rotation=45,
    )

    valid_ensemble_et_nse = extract_and_aggregate_metrics(inds_ensemble_et_valid, "NSE")
    plot_multi_single_comp_flow_boxes(
        valid_ensemble_et_nse,
        cases_exps_legends_together=cases_exps_legends_together,
        save_path=os.path.join(
            figure_dir,
            "mtl_valid_et_boxes_ensemble.png",
        ),
        rotation=45,
    )

    # -- Plot for testing period. 1/3 seems best as it didn't change make any variable's prediction worse.--
    chosen_idx = 3
    FIGURE_DPI = 600
    show_inds = ["Bias", "RMSE", "Corr", "NSE", "KGE"]
    q_metrices_results_chosen = extract_metrics_from_dict(
        inds_ensemble_q_test, [0, chosen_idx], show_inds
    )
    et_metrics_results_chosen = extract_metrics_from_dict(
        inds_ensemble_et_test, [0, chosen_idx], show_inds
    )
    chosen_metric_idx = 3
    # --------------- Plot all metrics for testing period ---------------------------
    # plot all metrics for stl and mtl exps
    # for Q
    # plot_boxes_matplotlib(
    #     q_metrices_results_chosen,
    #     label1=show_inds,
    #     label2=["STL", "MTL"],
    #     colorlst=["#d62728", "#1f77b4"],
    #     figsize=(10, 5),
    #     subplots_adjust_wspace=0.35,
    #     median_font_size="xx-small",
    # )
    # plt.savefig(
    #     os.path.join(
    #         figure_dir,
    #         "mtl_flow_test_all_metrices_boxes_ensemble.png",
    #     ),
    #     dpi=FIGURE_DPI,
    #     bbox_inches="tight",
    # )
    # # for ET
    # plot_boxes_matplotlib(
    #     et_metrics_results_chosen,
    #     label1=show_inds,
    #     label2=["STL", "MTL"],
    #     colorlst=["#d62728", "#1f77b4"],
    #     figsize=(10, 5),
    #     subplots_adjust_wspace=0.35,
    #     median_font_size="xx-small",
    # )
    # plt.savefig(
    #     os.path.join(
    #         figure_dir,
    #         "mtl_et_test_all_metrices_boxes_ensemble.png",
    #     ),
    #     dpi=FIGURE_DPI,
    #     bbox_inches="tight",
    # )
    # plot scatter with a 1:1 line to compare single-task and multi-task models
    plot_scatter(
        figure_dir,
        q_metrices_results_chosen[chosen_metric_idx][0],
        et_metrics_results_chosen[chosen_metric_idx][0],
        q_metrices_results_chosen[chosen_metric_idx][1],
        et_metrics_results_chosen[chosen_metric_idx][1],
        random_seed="ensemble",
        points_num=2,
        one_plot=True,
    )

    # ---- Plot time-series for some specific basins ------
    # plot_ts_figures(
    #     figure_dir,
    #     q_metrices_results_chosen[chosen_metric_idx][0],
    #     q_metrices_results_chosen[chosen_metric_idx][1],
    #     np.array(preds_q_test_ensemble),
    #     np.array(obss_q_test_ensemble),
    #     np.array(preds_et_test_ensemble),
    #     np.array(obss_et_test_ensemble),
    #     np.array(preds_q_train_ensemble),
    #     np.array(obss_q_train_ensemble),
    #     np.array(preds_et_train_ensemble),
    #     np.array(obss_et_train_ensemble),
    #     chosen_idx,
    #     random_seed="ensemble",
    #     points_num=2,
    # )

    # ----------------------  Plot maps -------------------------
    # plot map
    # plot_map_figures(
    #     figure_dir,
    #     q_metrices_results_chosen[chosen_metric_idx][0],
    #     et_metrics_results_chosen[chosen_metric_idx][0],
    #     q_metrices_results_chosen[chosen_metric_idx][1],
    #     et_metrics_results_chosen[chosen_metric_idx][1],
    #     random_seed="ensemble",
    #     points_num=2,
    # )
