"""
Author: Wenyu Ouyang
Date: 2022-11-21 15:53:23
LastEditTime: 2023-05-17 16:23:12
LastEditors: Wenyu Ouyang
Description: Train and test a linear probe for DL models
FilePath: /HydroMTL/hydromtl/explain/probe_analysis.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import numpy as np
import pandas as pd
import xarray as xr
import torch
from matplotlib import pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions
from hydromtl.utils.hydro_constant import HydroVar
from hydromtl.data.source import data_constant
from hydromtl.explain.cell_state_model import (
    get_all_models_weights,
    train_model_loop,
    calculate_predictions,
    calculate_raw_correlations,
    LinearModel,
)
from hydromtl.explain.explain_lstm import (
    calculate_all_error_metrics,
    get_input_target_data_for_corr_analysis,
)
from hydromtl.models.training_utils import get_the_device


def train_probe(run_exp, var="ET", retrain=False, **kwargs):
    """train a linear probe model for an experiment

    refer to: https://github.com/tommylees112/neuralhydrology/tree/pixel/notebooks

    Parameters
    ----------
    run_exp : str
        experiment name, such as camels/exp41013
    var : str, optional
        _description_, by default "ET"
    retrain : bool, optional
        whether to retrain the probe model, by default False
    kwargs : dict
        other parameters for training the probe model
        key-values include: {
            "train_test": True,
            "train_val": False,
            "num_workers": 4,
            "device": -1,
            "drop_out": 0.0,
        }
    """
    # we only support MODIS ET, usgsFlow and SMAP ssm now
    assert var in ["ET", "usgsFlow", "ssm"]
    exp = run_exp.split(os.sep)
    run_dir = os.path.join(definitions.RESULT_DIR, exp[0], exp[1])
    input_data, target_data = get_input_target_data_for_corr_analysis(run_exp, var)
    #  calculate raw correlations (cell state and values)
    print("-- Running RAW Correlations --")
    all_corrs_file = os.path.join(run_dir, "all_corrs_cs_" + var + ".npy")
    all_basin_corrs_file = os.path.join(run_dir, "all_basin_corrs_cs_" + var + ".nc")
    if os.path.exists(all_corrs_file) and os.path.exists(all_basin_corrs_file):
        all_corrs = np.load(all_corrs_file)
        all_basin_corrs = xr.open_dataarray(all_basin_corrs_file)
    else:
        all_corrs, all_basin_corrs = calculate_raw_correlations(input_data, target_data)
        np.save(all_corrs_file, all_corrs)
        all_basin_corrs.to_netcdf(all_basin_corrs_file)
    start_date = pd.to_datetime(input_data.time.min().values)
    end_date = pd.to_datetime(input_data.time.max().values)
    probe_train_params = {
        "train_test": True,
        "train_val": False,
        "num_workers": 4,
        "device": -1,
        "drop_out": 0.0,
    }
    probe_train_params.update(kwargs)
    device = get_the_device(probe_train_params["device"])
    drop_out = probe_train_params["drop_out"]

    print("-- Get Probe Predictions --")
    linear_probe_preds_file = os.path.join(run_dir, "linear_probe_preds_" + var + ".nc")
    linear_probe_model_file = os.path.join(
        run_dir, "linear_probe_model_" + var + ".pth"
    )
    if (
        (not retrain)
        and os.path.exists(linear_probe_preds_file)
        and os.path.exists(linear_probe_model_file)
    ):
        preds = xr.open_dataset(linear_probe_preds_file)
        checkpoint = torch.load(linear_probe_model_file)
        d_in = len(input_data.dimension.values)
        model = LinearModel(D_in=d_in, dropout=drop_out)
        model.load_state_dict(checkpoint, strict=True)
    else:
        print("-- Training Model for " + var + " --")
        train_losses, model, test_loader = train_model_loop(
            input_data=input_data,
            target_data=target_data,  #  needs to be xr.DataArray
            train_test=probe_train_params["train_test"],
            train_val=probe_train_params["train_val"],
            return_loaders=True,
            start_date=start_date,
            end_date=end_date,
            num_workers=probe_train_params["num_workers"],
            l2_penalty=2,
            device=device,
            dropout=drop_out,
        )
        torch.save(model.state_dict(), linear_probe_model_file)
        #  run forward pass and convert to xarray object
        print("-- Running Test-set Predictions --")
        preds = calculate_predictions(model, test_loader, device)
        preds.to_netcdf(linear_probe_preds_file)

    print("-- Get metrics of probe predictions --")
    linear_probe_error_file = os.path.join(run_dir, "linear_probe_error_" + var + ".nc")
    if os.path.exists(linear_probe_error_file):
        errors = xr.open_dataset(linear_probe_error_file)
    else:
        errors = calculate_all_error_metrics(
            preds,
            basin_coord="station_id",
            obs_var="y",
            sim_var="y_hat",
            time_coord="time",
        )
        errors.to_netcdf(linear_probe_error_file)

    # extract weights and biases
    print("-- Extracting probe's weights and biases --")
    linear_probe_weights_file = os.path.join(
        run_dir, "linear_probe_weights_" + var + ".npy"
    )
    linear_probe_biases_file = os.path.join(
        run_dir, "linear_probe_biases_" + var + ".npy"
    )
    if os.path.exists(linear_probe_weights_file) and os.path.exists(
        linear_probe_biases_file
    ):
        ws = np.load(linear_probe_weights_file)
        bs = np.load(linear_probe_biases_file)
    else:
        ws, bs = get_all_models_weights([model])
        np.save(linear_probe_weights_file, ws)
        np.save(linear_probe_biases_file, bs)
    return all_corrs, all_basin_corrs, errors, ws, bs


def plot_one_probe(run_exp, var: HydroVar, all_basin_corrs, ws, save_dir=None):
    exp = run_exp.split(os.sep)
    run_dir = os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", exp[0], exp[1])
    if save_dir is None:
        save_dir = run_dir
    FIGURE_DPI = 600
    print(
        "-- Plotting cs~"
        + var.name
        + " basin-correlations and probe weights for "
        + run_exp
        + " --"
    )
    fig, ax1 = plt.subplots()
    all_basin_corrs.plot(ax=ax1, cmap="RdBu_r")
    # https://docs.xarray.dev/en/stable/user-guide/plotting.html
    plt.xlabel("Cell", fontsize=16)
    plt.ylabel("Basin", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # ax1.set_title("单元状态与" + var.ChineseName + "相关性", fontsize=16)
    ax1.set_title("Correlation between cell state and " + var.name + " for all basins")
    plt.savefig(
        os.path.join(
            save_dir, exp[0] + "_" + exp[1] + "_all_basin_corrs_" + var.name + ".png"
        ),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    f, ax2 = plt.subplots(figsize=(12, 2))
    im = ax2.pcolormesh(ws)
    # ax2.set_ylabel(var.ChineseName, fontsize=16)
    ax2.set_ylabel(var.name, fontsize=16)
    # ax2.set_title(var.ChineseName + "线性探测器的权重（所有单元状态）", fontsize=16)
    ax2.set_title("Weights of linear probe for " + var.name + " (all cell states)")
    plt.colorbar(im, orientation="horizontal")
    plt.savefig(
        os.path.join(
            save_dir, exp[0] + "_" + exp[1] + "_linear_probe_ws_" + var.name + ".png"
        ),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )


def show_probe(
    run_exp_lst,
    var: HydroVar,
    legend_lst=None,
    show_probe_metric="Corr",
    retrian_probe=None,
    save_dir=definitions.RESULT_DIR,
    **kwargs,
):
    """Show all probe results for a list of experiments

    Parameters
    ----------
    run_exp_lst : list
        list of experiment names, such as ["camels/exp41001", "camels/exp41013"]
    var : HydroVar
        variable to show
    legend_lst : list, optional
        list of legend names, by default ["STL", "MTL"]
    show_probe_metric : str, optional
        metric to show, by default "Corr"
    retrian_probe : list, optional
        whether to retrain the probe, by default [False, False]
    kwargs : dict
        other arguments for probe training parameters
        key-values include:
        {
            "train_test": True,
            "train_val": False,
            "num_workers": 4,
            "device": -1,  # -1 mean cpu, 0, 1, .. mean different gpu number
            "drop_out": 0.0,
        }
    """
    if legend_lst is None:
        legend_lst = ["STL", "MTL"]
    if retrian_probe is None:
        retrian_probe = [False, False]
    # save compared results in the directory of first experiment
    all_corrs_lst = []
    all_basin_corrs_lst = []
    errors_lst = []
    ws_lst = []
    bs_lst = []
    for i, run_exp in enumerate(run_exp_lst):
        all_corrs, all_basin_corrs, errors, ws, bs = train_probe(
            run_exp=run_exp, var=var.name, retrain=retrian_probe[i], **kwargs
        )
        all_corrs_lst.append(all_corrs)
        all_basin_corrs_lst.append(all_basin_corrs)
        errors_lst.append(errors)
        ws_lst.append(ws)
        bs_lst.append(bs)
        plot_one_probe(run_exp, var, all_basin_corrs, ws, save_dir=save_dir)
    FIGURE_DPI = 600
    print(
        "-- Comparing cs~" + var.name + " correlations for " + str(run_exp_lst) + " --"
    )
    f, ax1 = plt.subplots(figsize=(12, 4))
    for i, run_exp in enumerate(run_exp_lst):
        ax1.hist(
            all_corrs_lst[i],
            alpha=0.6,
            bins=100,
            label=legend_lst[i],
            color=f"C{str(i)}",
        )
        # ax1.axvline(np.median(all_corrs_lst[i]), ls="--", color=f"C{str(i)}")
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
    ax1.legend()
    # ax1.set_title("各模型所有流域单元状态与" + var.ChineseName + "的相关性分布", fontsize=16)
    ax1.set_title(
        "Correlation between one cell state of all basins and the observation of "
        + var.name
    )
    ax1.set_xlabel("Correlation")
    # ax1.set_xlabel("相关系数", fontsize=16)
    ax1.set_ylabel("Frequency", fontsize=16)
    # ax1.set_ylabel("频数", fontsize=16)
    sns.despine()
    plt.savefig(
        os.path.join(save_dir, "all_corrs_" + var.name + ".png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    print(
        "-- Comparing cs~"
        + var.name
        + " probe prediction metric "
        + show_probe_metric
        + " for "
        + str(run_exp_lst)
        + " --"
    )
    f, ax2 = plt.subplots(figsize=(12, 4))
    for i, run_exp in enumerate(run_exp_lst):
        ax2.hist(
            errors_lst[i][show_probe_metric],
            alpha=0.6,
            bins=100,
            label=legend_lst[i],
            color=f"C{str(i)}",
        )
        ax2.axvline(
            np.median(errors_lst[i][show_probe_metric]),
            ls="--",
            color=f"C{str(i)}",
        )
    ax2.legend()

    ax2.set_xlabel(show_probe_metric, fontsize=16)
    # ax2.set_title(var.ChineseName + "探测器预测" + show_probe_metric + "分布", fontsize=16)
    ax2.set_title(
        show_probe_metric
        + " between the prediction of probe and the observation for "
        + var.name
    )
    ax2.set_ylabel("Frequency", fontsize=16)
    # ax2.set_ylabel("频数", fontsize=16)

    sns.despine()
    plt.savefig(
        os.path.join(
            save_dir,
            "linear_probe_preds_" + var.name + "_" + show_probe_metric + ".png",
        ),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    run_exp_lst = [
        "camels" + os.sep + "exp410010",
        "camels" + os.sep + "exp410130",
        "camels" + os.sep + "exp42001",
    ]
    show_probe(
        run_exp_lst=run_exp_lst,
        var=data_constant.surface_soil_moisture_smap_camels_us,
        legend_lst=["STL-Q", "MTL", "STL-ET"],
        show_probe_metric="Corr",
        retrian_probe=[False, False, False],
        num_workers=4,
    )
