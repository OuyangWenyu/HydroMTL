"""
Author: Wenyu Ouyang
Date: 2022-01-08 16:58:14
LastEditTime: 2024-05-19 14:20:38
LastEditors: Wenyu Ouyang
Description: Choose some basins for training and testing of multioutput exps
FilePath: \HydroMTL\scripts\prepare_data.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from hydromtl.utils import hydro_stat, hydro_utils
from hydromtl.data.source_pro.select_gages_ids import (
    usgs_screen_streamflow,
)
from hydromtl.data.source.data_camels import Camels
from hydromtl.data.source_pro.data_camels_pro import CamelsPro
from hydromtl.visual.plot_stat import plot_ts
from scripts.app_constant import VAR_T_CHOSEN_FROM_NLDAS

ID_FILE_PATH = os.path.join(
    definitions.RESULT_DIR, "camels_us_mtl_2001_2021_flow_screen.csv"
)
AUGDATA_TRANGE = ["2001-10-01", "2021-10-01"]
COMP_TRANGE = ["2001-10-01", "2014-10-01"]
# set font
plt.rcParams["font.family"] = "Times New Roman"


def select_basins():
    # to avoid the effect from bad-quality streamflow data, we exclude sites with over 5% (about 1 year) nan values
    camels_dir = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
    camels = Camels(camels_dir)
    flow_screen_param = {"missing_data_ratio": 0.05}
    selected_ids = usgs_screen_streamflow(
        camels,
        camels.read_object_ids().tolist(),
        AUGDATA_TRANGE,
        **flow_screen_param,
    )
    df_camels_mtl = pd.DataFrame({"GAGE_ID": selected_ids})
    df_camels_mtl.to_csv(
        ID_FILE_PATH,
        quoting=csv.QUOTE_NONNUMERIC,
        index=None,
    )


def compare_nldas():
    if not os.path.exists(ID_FILE_PATH):
        select_basins()
    nldas_camels_file = os.path.join(definitions.RESULT_DIR, "nldas_camels.npy")
    nldas_gee_file = os.path.join(definitions.RESULT_DIR, "nldas_madeingee.npy")
    if os.path.exists(nldas_camels_file):
        nldas_camels = np.load(nldas_camels_file)
    if os.path.exists(nldas_gee_file):
        nldas_madeingee = np.load(nldas_gee_file)
    if (not os.path.exists(nldas_camels_file)) and (not os.path.exists(nldas_gee_file)):
        nldas_camels, nldas_madeingee = _read_nldas_data(
            nldas_camels_file, nldas_gee_file
        )
    # compare the nldas data from camels and madeingee
    # all are numpy arrays
    metrics = hydro_stat.stat_error(
        nldas_camels, nldas_madeingee, fill_nan=["no", "no"]
    )
    print(metrics)


def _read_nldas_data(nldas_camels_file, nldas_gee_file):
    source_path = [
        os.path.join(definitions.DATASET_DIR, "camelsflowet"),
        os.path.join(definitions.DATASET_DIR, "modiset4camels"),
        os.path.join(definitions.DATASET_DIR, "camels", "camels_us"),
        os.path.join(definitions.DATASET_DIR, "nldas4camels"),
        os.path.join(definitions.DATASET_DIR, "smap4camels"),
    ]
    camels_pro = CamelsPro(source_path)
    camels_dir = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
    camels = Camels(camels_dir)
    basin_ids = pd.read_csv(ID_FILE_PATH, dtype={"GAGE_ID": str})["GAGE_ID"].tolist()
    nldas_camels = camels.read_relevant_cols(
        basin_ids,
        t_range=COMP_TRANGE,
        # only these two variables are included in both camels and GEE nldas data
        # unit: PRCP(mm/day)	SRAD(W/m2)
        var_lst=["prcp", "srad"],
        forcing_type="nldas",
    )
    np.save(nldas_camels_file, nldas_camels)
    nldas_madeingee = camels_pro.read_relevant_cols(
        basin_ids,
        t_range_list=COMP_TRANGE,
        # total_precipitation(kg/m^2) shortwave_radiation(W/m^2)
        relevant_cols=["total_precipitation", "shortwave_radiation"],
        forcing_type="nldas",
    )
    np.save(nldas_gee_file, nldas_madeingee)
    return nldas_camels, nldas_madeingee


def see_basin_area():
    camels_dir = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
    camels = Camels(camels_dir)
    if not os.path.exists(ID_FILE_PATH):
        select_basins()
    basin_ids = pd.read_csv(ID_FILE_PATH, dtype={"GAGE_ID": str})["GAGE_ID"].tolist()
    basin_areas = camels.read_basin_area(basin_ids)
    print(basin_areas)
    data = basin_areas.flatten()
    one_nldas_cell_area = 13915 * 13915 / 1e6
    greater_than_one_grid = np.sum(data >= one_nldas_cell_area)
    print(f"Basins with area greater than one NLDAS grid: {greater_than_one_grid}")
    greater_than_four_grids = np.sum(data >= (4 * one_nldas_cell_area))
    print(f"Basins with area greater than four NLDAS grids: {greater_than_four_grids}")

    # Calculate the threshold of 0.5% times the area of one NLDAS grid cell in square kilometers
    threshold_area = one_nldas_cell_area * 0.005
    # Find the indices of watersheds with area less than the threshold area
    indices_less_than_threshold = np.where(data < threshold_area)[0]
    indices_less_than_threshold.tolist()  # Convert to list for easier reading if necessary

    min_basin = basin_ids[np.argmin(data)]
    min_basin_area = np.min(data)
    print(f"Basin with minimum area: {min_basin} with area: {min_basin_area}")

    log_data = np.log(data)
    # Create a histogram with logarithmic data
    plt.figure(figsize=(10, 6))
    plt.hist(log_data, bins=30, color="blue", alpha=0.7)
    plt.title("Log-scaled Histogram of Watershed Areas")
    plt.xlabel("Log of Area")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def see_basin_streamflow_data():
    camels_dir = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
    camels = Camels(camels_dir)
    if not os.path.exists(ID_FILE_PATH):
        select_basins()
    all_basins = camels.read_object_ids()
    chosen_basins = pd.read_csv(ID_FILE_PATH, dtype={"GAGE_ID": str})[
        "GAGE_ID"
    ].tolist()
    # Use NumPy to find unchosen basins
    unchosen_basins = np.setdiff1d(all_basins, chosen_basins)
    unchosen_flow = camels.read_target_cols(
        unchosen_basins, t_range=AUGDATA_TRANGE, target_cols=["usgsFlow"]
    )
    # Calculate the missing data rate for each basin
    # Count the number of nan values for each basin along the time axis
    num_nans = np.isnan(unchosen_flow).sum(axis=1)  # Sum over the time dimension
    total_data_points = unchosen_flow.shape[1]  # Total number of time points

    # Calculate missing data rate as a percentage
    missing_data_rate = (num_nans / total_data_points) * 100

    # Find indices of the basin with the max and min missing data rates
    max_missing_idx = np.argmax(missing_data_rate)
    min_missing_idx = np.argmin(missing_data_rate)
    # Optionally, print the missing data rate for each basin
    print("Missing data rates for each basin (%):", missing_data_rate)
    # Plot the streamflow data of the first unchosen basin
    t_lst = hydro_utils.t_range_days(AUGDATA_TRANGE)
    figure_dir = os.path.join(definitions.RESULT_DIR, "figures", "prepare_data")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    plot_ts(
        t_lst,
        unchosen_flow[max_missing_idx, :, 0],
        title=f"Time Series for Basin {unchosen_basins[max_missing_idx]} with Maximum Missing Data",
        xlabel="Date",
        ylabel="Streamflow(m$^3$/s)",
    )
    plt.savefig(
        os.path.join(figure_dir, "max_missing_data_basin_flow_ts.png"),
        dpi=600,
    )
    plot_ts(
        t_lst,
        unchosen_flow[min_missing_idx, :, 0],
        title=f"Time Series for Basin {unchosen_basins[min_missing_idx]} with Minimum Missing Data",
        xlabel="Date",
        ylabel="Streamflow(m$^3$/s)",
    )
    plt.savefig(
        os.path.join(figure_dir, "min_missing_data_basin_flow_ts.png"),
        dpi=600,
    )


if __name__ == "__main__":
    # select_basins()
    # compare_nldas()
    # see_basin_area()
    see_basin_streamflow_data()
