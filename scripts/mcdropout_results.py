"""
Author: Wenyu Ouyang
Date: 2024-10-10 18:25:22
LastEditTime: 2024-10-15 21:28:52
LastEditors: Wenyu Ouyang
Description: Run all cases, save the results and plot the results
FilePath: \HydroMTL\scripts\mcdropout_results.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np


project_dir = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(project_dir)
from definitions import RESULT_DIR
from hydromtl.explain.uncertainty_analysis import (
    aggregate_and_plot_calibration,
    calculate_error_exceedance_prob,
    plot_calibration_curve,
    plot_probability_plot,
    pp_plot_multiple_basins,
    process_and_aggregate_basins,
)
from scripts.mcdropout_eval import run_mcdropout_for_configs


def read_mcdropout_results(exp):
    # read the results
    pred_file = os.path.join(
        RESULT_DIR,
        "camels",
        exp,
        "epoch300flow_pred.npy",
    )
    obs_file = os.path.join(
        RESULT_DIR,
        "camels",
        exp,
        "epoch300flow_obs.npy",
    )
    preds = np.load(pred_file)
    # calculate the mean and std of the predictions
    pred_mean = np.mean(preds, axis=0)
    pred_std = np.std(preds, axis=0)
    obs = np.load(obs_file)
    return preds, obs, pred_mean, pred_std


run_mode = False
# each has 5 random seeds: 1234, 12345, 123, 111, 1111
exp_q_lst = [
    "expstlq001",
    "expstlq202",
    "expstlq203",
    "expstlq204",
    "expstlq205",
]
exp_et_lst = [
    "expstlet001",
    "expstlet003",
    "expstlet002",
    "expstlet004",
    "expstlet005",
]
exp_mtl_lst = [
    "expmtl003",
    "expmtl203",
    "expmtl303",
    "expmtl403",
    "expmtl503",
]
if run_mode:
    run_mcdropout_for_configs(exp_q_lst, 100, post_fix=2)
    run_mcdropout_for_configs(exp_et_lst, 100, post_fix=2)
    run_mcdropout_for_configs(exp_mtl_lst, 100, post_fix=2)

# read the results and plot
post_fix = 1
exp_q_lst_uncertainty = [f"{exp}0{post_fix}" for exp in exp_q_lst]
exp_et_lst_uncertainty = [f"{exp}0{post_fix}" for exp in exp_et_lst]
exp_mtl_lst_uncertainty = [f"{exp}0{post_fix}" for exp in exp_mtl_lst]
exp_lst = exp_q_lst_uncertainty + exp_et_lst_uncertainty + exp_mtl_lst_uncertainty
# exp_lst = exp_mtl_lst_uncertainty
for exp in exp_lst:
    print(f"Plotting {exp}")
    preds, obs, pred_mean, pred_std = read_mcdropout_results(exp)
    cc_save_path = os.path.join(RESULT_DIR, f"{exp}_calibration_curve.png")
    aggregate_and_plot_calibration(
        obs[:, :, 0], preds[:, :, :, 0], save_path=cc_save_path
    )
    num_basins = obs.shape[0]
    basins_data = []
    for i in range(num_basins - 5, num_basins):
        # Randomly generate observed values and predictions for each basin
        basin_name = f"Basin {i+1}"
        # Store the data in the list
        basins_data.append(
            {
                # 0 means the first output, which is the flow
                "predictions": preds[:, i, :, 0],
                "obs_values": obs[i, :, 0],
                "name": basin_name,
            }
        )
    # Aggregate z-values and r-values over basins and time steps
    pp_plot_multiple_basins(basins_data)
    all_z_values, all_r_values = process_and_aggregate_basins(basins_data, num_bins=500)
    pp_save_path = os.path.join(RESULT_DIR, f"{exp}_probability_plot.png")
    # Plot the aggregated probability plot
    plot_probability_plot(
        all_z_values,
        all_r_values,
        basin_name="All Basins",
        save_path=pp_save_path,
    )
    plt.show()
