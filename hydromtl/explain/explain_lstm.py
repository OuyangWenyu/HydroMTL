"""
Author: Wenyu Ouyang
Date: 2022-11-19 21:05:32
LastEditTime: 2023-05-17 09:22:36
LastEditors: Wenyu Ouyang
Description: Extract information from LSTM
FilePath: /HydroMTL/hydromtl/explain/explain_lstm.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from tqdm import tqdm
from typing import List, Optional, Dict
import pandas as pd
import xarray as xr
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions
from hydromtl.explain.read_lstm_results import (
    load_ts_var_data_for_exp,
    normalize_xr_by_basin,
    load_cell_states_for_exp,
    normalize_xarray_cstate,
)
from hydromtl.explain.cell_state_dataset import (
    fill_gaps,
    choose_data_like_et_or_ssm,
)
from hydromtl.utils.hydro_exceptions import AllNaNError
from hydromtl.utils import hydro_stat


def get_input_target_data_for_corr_analysis(run_exp, var="ET"):
    """Get input and target data for correlation analysis

    Parameters
    ----------
    run_exp : str
        the exp id and name
    var : str, optional
        the variable name, by default "ET"
    """
    # we only support MODIS ET, usgsFlow and SMAP ssm now
    assert var in ["ET", "usgsFlow", "ssm"]
    if var == "ET" or var == "ssm":
        fill_nan = "et_ssm_ignore"
    else:
        fill_nan = "interpolate"
    exp = run_exp.split(os.sep)
    run_dir = os.path.join(definitions.RESULT_DIR, exp[0], exp[-1])
    input_data_nc_file = os.path.join(run_dir, "corr_analysis_input_data_cs.nc")
    target_data_nc_file = os.path.join(
        run_dir, "corr_analysis_target_data_" + var + ".nc"
    )
    if os.path.exists(input_data_nc_file) and os.path.exists(target_data_nc_file):
        input_data = xr.open_dataarray(input_data_nc_file)
        target_data = xr.open_dataarray(target_data_nc_file)
        return input_data, target_data
    obs = load_ts_var_data_for_exp(run_exp, var)
    norm_obs = normalize_xr_by_basin(obs)
    norm_obs_fillgap = fill_gaps(norm_obs, fill_nan=[fill_nan])
    cn = load_cell_states_for_exp(run_exp)
    cs_data = cn.copy()
    norm_cstate = normalize_xarray_cstate(
        cs_data,
        cell_state_var="c_n",
        cell_states_nc_file=os.path.join(run_dir, "normed_cell_states.nc"),
    )
    if var == "ET" or var == "ssm":
        input_data, target_data = choose_data_like_et_or_ssm(
            norm_cstate["c_n"], norm_obs_fillgap["obs"].sel(dimension=var), var
        )
    elif var == "usgsFlow":
        input_data = norm_cstate["c_n"]
        target_data = norm_obs_fillgap["obs"].sel(dimension=var)
    else:
        raise ValueError("var must be ET, usgsFlow or ssm")

    # TODO: unify date col name
    input_data = input_data.rename({"date": "time"})
    target_data = target_data.rename({"date": "time"})
    input_data.to_netcdf(input_data_nc_file)
    target_data.to_netcdf(target_data_nc_file)
    return input_data, target_data


def _check_all_nan(obs: xr.DataArray, sim: xr.DataArray):
    """Check if all observations or simulations are NaN and raise an exception if this is the case.

    Raises
    ------
    AllNaNError
        If all observations or all simulations are NaN.
    """
    if all(obs.isnull()):
        raise AllNaNError("All observed values are NaN, thus metrics will be NaN, too.")
    if all(sim.isnull()):
        raise AllNaNError(
            "All simulated values are NaN, thus metrics will be NaN, too."
        )


def calculate_all_metrics(obs: xr.DataArray, sim: xr.DataArray) -> Dict[str, float]:
    """Calculate all metrics with default values.

    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys corresponding to metric name and values corresponding to metric values.

    Raises
    ------
    AllNaNError
        If all observations or all simulations are NaN.
    """
    _check_all_nan(obs, sim)
    return (
        hydro_stat.stat_error(obs.values, sim.values)
        if len(obs.dims) > 1
        else hydro_stat.stat_error_i(obs.values, sim.values)
    )


def _check_metrics(metrics: Optional[List[str]]) -> None:
    if metrics is not None:
        all_metrics = hydro_stat.ALL_METRICS
        assert all(
            m in all_metrics for m in metrics
        ), f"Metrics must be one of {all_metrics}. You provided: {metrics}"


def calculate_all_error_metrics(
    preds: xr.Dataset,
    basin_coord: str = "basin",
    time_coord: str = "date",
    obs_var: str = "streamflow_obs",
    sim_var: str = "streamflow_sim",
    metrics: Optional[List[str]] = None,
) -> xr.Dataset:
    all_errors: List[pd.DataFrame] = []
    missing_data: List[str] = []

    _check_metrics(metrics)

    pbar = tqdm(preds[basin_coord].values, desc="Calculating Errors")
    for sid in pbar:
        pbar.set_postfix_str(sid)
        sim = (
            preds[sim_var]
            .rename({basin_coord: "station_id", time_coord: "date"})
            .sel(station_id=sid)
        )
        obs = (
            preds[obs_var]
            .rename({basin_coord: "station_id", time_coord: "date"})
            .sel(station_id=sid)
        )

        try:
            errors = calculate_all_metrics(sim=sim, obs=obs)
            all_errors.append(pd.DataFrame({sid: errors}).T)
        except AllNaNError:
            missing_data.append(sid)

    errors = pd.concat(all_errors).to_xarray().rename({"index": basin_coord})
    return errors
