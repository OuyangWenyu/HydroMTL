"""
Author: Wenyu Ouyang
Date: 2024-05-14 17:44:31
LastEditTime: 2024-05-29 12:55:39
LastEditors: Wenyu Ouyang
Description: scripts same with assess_reliability.ipynb
FilePath: \HydroMTL\scripts\assess_ensemble_reliability.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
import xarray as xr


# Get the current directory of the project
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from hydromtl.data.source import data_constant
from hydromtl.explain.explain_lstm import calculate_all_error_metrics
from hydromtl.explain.probe_analysis import plot_errors, show_probe

random_seed = [1234, 12345, 123, 111, 1111]
run_exp_lst = [
    [
        f"camels{os.sep}expstlq0010",
        f"camels{os.sep}expmtl0030",
        f"camels{os.sep}expstlet0010",
    ],
    [
        f"camels{os.sep}expstlq2020",
        f"camels{os.sep}expmtl2030",
        f"camels{os.sep}expstlet0030",
    ],
    [
        f"camels{os.sep}expstlq2030",
        f"camels{os.sep}expmtl3030",
        f"camels{os.sep}expstlet0020",
    ],
    [
        f"camels{os.sep}expstlq2040",
        f"camels{os.sep}expmtl4030",
        f"camels{os.sep}expstlet0040",
    ],
    [
        f"camels{os.sep}expstlq2050",
        f"camels{os.sep}expmtl5030",
        f"camels{os.sep}expstlet0050",
    ],
]
save_dir = os.path.join(definitions.RESULT_DIR, "figures", "ensemble")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
legend_lst = ["STL-Q", "MTL", "STL-ET"]


def compute_mean_dataarrays(data_list):
    """
    Computes the mean of corresponding DataArrays in nested sublists for specified variables.

    Args:
        data_list (list of list of xarray.DataArray): A list containing sublists of DataArrays.

    Returns:
        list: A list of xarray.DataArray, each representing the mean of DataArrays at corresponding positions.
    """
    num_positions = len(
        data_list[0]
    )  # Assumes all sublists have the same number of DataArrays
    combined_means = []

    # Iterate over each position in the sublists
    for i in range(num_positions):
        # Collect all DataArrays at the current position
        arrays_at_position = [sublist[i] for sublist in data_list if i < len(sublist)]

        # Stack these DataArrays along a new dimension
        stacked_array = xr.concat(arrays_at_position, dim="new_dim")

        # Compute the mean along the new dimension
        mean_array = stacked_array.mean(dim="new_dim")

        # Append the resulting DataArray to the result list
        combined_means.append(mean_array)

    return combined_means


def show_probe_ensemble(
    run_exp_lst,
    var,
    legend_lst,
    show_probe_metric="Corr",
    save_dir=None,
):
    preds_all_lst = []
    run_exp_ensemble_lst = [
        f"camels{os.sep}expstlqensemble",
        f"camels{os.sep}expmtl3to1ensemble",
        f"camels{os.sep}expstletensemble",
    ]
    for i in range(len(run_exp_lst)):
        run_exp = run_exp_lst[i]
        preds = show_probe(
            run_exp,
            var=var,
            legend_lst=legend_lst,
            show_probe_metric=show_probe_metric,
            retrian_probe=[False, False, False],
            num_workers=0,
            save_dir=save_dir,
            is_plot=False,
        )
        preds_all_lst.append(preds)
    preds_ensemble_lst = compute_mean_dataarrays(preds_all_lst)
    errors_ensemble_lst = []
    for i in range(len(preds_ensemble_lst)):
        linear_probe_error_file = os.path.join(
            save_dir,
            f"linear_probe_error_{var.name}_{run_exp_ensemble_lst[i].split(os.sep)[-1]}.nc",
        )
        if os.path.exists(linear_probe_error_file):
            errors_ensemble = xr.open_dataset(linear_probe_error_file)
        else:
            errors_ensemble = calculate_all_error_metrics(
                preds_ensemble_lst[i],
                basin_coord="station_id",
                obs_var="y",
                sim_var="y_hat",
                time_coord="time",
            )
            errors_ensemble.to_netcdf(linear_probe_error_file)
        errors_ensemble_lst.append(errors_ensemble)

    plot_errors(
        run_exp_ensemble_lst,
        var,
        legend_lst,
        show_probe_metric,
        save_dir,
        "state",
        errors_ensemble_lst,
    )


# First probe is for evapotranspiration (ET).
show_probe_ensemble(
    run_exp_lst=run_exp_lst,
    var=data_constant.evapotranspiration_modis_camels_us,
    legend_lst=legend_lst,
    show_probe_metric="Corr",
    save_dir=save_dir,
)
# The second probe is for streamflow (Q).
show_probe_ensemble(
    run_exp_lst=run_exp_lst,
    var=data_constant.streamflow_camels_us,
    legend_lst=legend_lst,
    show_probe_metric="Corr",
    save_dir=save_dir,
)
# The final probe is for soil moisture (SM).
show_probe_ensemble(
    run_exp_lst=run_exp_lst,
    var=data_constant.surface_soil_moisture_smap_camels_us,
    legend_lst=legend_lst,
    show_probe_metric="Corr",
    save_dir=save_dir,
)
