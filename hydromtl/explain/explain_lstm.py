"""
Author: Wenyu Ouyang
Date: 2022-11-19 21:05:32
LastEditTime: 2024-04-17 09:35:26
LastEditors: Wenyu Ouyang
Description: Extract information from LSTM
FilePath: \HydroMTL\hydromtl\explain\explain_lstm.py
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
from app_constant import VAR_T_CHOSEN_FROM_NLDAS
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


abbreviations = {
    "total_precipitation": "tp",
    "potential_evaporation": "pet",
    "temperature": "tmp",
    "specific_humidity": "sh",
    "shortwave_radiation": "swr",
    "potential_energy": "pe",
}


def abbreviate_and_join(variables):
    """
    Abbreviates the given list of variables and joins them with underscores.

    Parameters:
    - variables (list): A list of variable names in full form.

    Returns:
    - str: A string of the abbreviated variable names joined by underscores.
    """
    # Map the full variable names to their abbreviations
    abbreviations_list = [
        abbreviations[var] for var in variables if var in abbreviations
    ]

    # Join the abbreviations with underscores
    return "_".join(abbreviations_list)


def _get_input_target_data_for_corr_analysis(run_exp, var="ET", probe_input="state"):
    """Get input and target data for correlation analysis

    Parameters
    ----------
    run_exp : str
        the exp id and name
    var : str, optional
        the variable name, by default "ET"
    probe_input : str, optional
        must be "state" or "output", by default "state"
    """
    # we only support MODIS ET, usgsFlow and SMAP ssm now
    assert var in ["ET", "usgsFlow", "ssm"]
    fillnan4output = "et_ssm_ignore" if var in ["ET", "ssm"] else "interpolate"
    exp = run_exp.split(os.sep)
    run_dir = os.path.join(definitions.RESULT_DIR, exp[0], exp[-1])
    input_data_nc_file = os.path.join(run_dir, "corr_analysis_input_data_cs.nc")
    if probe_input != "state" and set(probe_input).issubset(
        set(VAR_T_CHOSEN_FROM_NLDAS)
    ):
        input_data_nc_file = os.path.join(
            run_dir, f"corr_analysis_input_data_{abbreviate_and_join(probe_input)}.nc"
        )
    target_data_nc_file = os.path.join(
        run_dir, "corr_analysis_target_data_" + var + ".nc"
    )
    if os.path.exists(input_data_nc_file) and os.path.exists(target_data_nc_file):
        input_data = xr.open_dataarray(input_data_nc_file)
        target_data = xr.open_dataarray(target_data_nc_file)
        return input_data, target_data
    if probe_input == "state":
        cn = load_cell_states_for_exp(run_exp)
        cs_data = cn.copy()
        norm_input = normalize_xarray_cstate(
            cs_data,
            cell_state_var="c_n",
            cell_states_nc_file=os.path.join(run_dir, "normed_cell_states.nc"),
        )
    else:
        fillnan4forcing = "interpolate"  # ['interpolate' for _ in probe_input]
        norm_input = _get_norm_ts_data(run_exp, probe_input, fillnan4forcing)

    norm_obs_fillgap = _get_norm_ts_data(run_exp, var, fillnan4output)

    if probe_input == "state":
        input_ = norm_input["c_n"]
    else:
        # keep the dimension of forcing variable, using [probe_input]
        input_ = norm_input["forcing"].sel(dimension=probe_input)
    if var in ["ET", "ssm"]:
        input_data, target_data = choose_data_like_et_or_ssm(
            input_, norm_obs_fillgap["obs"].sel(dimension=var), var
        )
    elif var == "usgsFlow":
        input_data = input_
        target_data = norm_obs_fillgap["obs"].sel(dimension=var)
    else:
        raise ValueError("var must be ET, usgsFlow or ssm")

    # TODO: unify date col name
    input_data = input_data.rename({"date": "time"})
    target_data = target_data.rename({"date": "time"})
    input_data.to_netcdf(input_data_nc_file)
    target_data.to_netcdf(target_data_nc_file)
    return input_data, target_data


# TODO Rename this here and in `_get_input_target_data_for_corr_analysis`
def _get_norm_ts_data(run_exp, vars, fill_nan):
    ts_data = load_ts_var_data_for_exp(run_exp, vars)
    norm_input_ = normalize_xr_by_basin(ts_data)
    if type(fill_nan) == str:
        if type(vars) == list and len(vars) > 1:
            fill_nan = [fill_nan for _ in vars]
        else:
            fill_nan = [fill_nan]
    return fill_gaps(norm_input_, fill_nan=fill_nan)


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
