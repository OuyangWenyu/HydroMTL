"""
Author: Wenyu Ouyang
Date: 2022-01-08 17:31:35
LastEditTime: 2024-05-10 14:20:43
LastEditors: Wenyu Ouyang
Description: Some util functions for scripts in app/streamflow
FilePath: \HydroMTL\scripts\streamflow_utils.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import csv
import shutil
from functools import reduce
import os
import sys
from pathlib import Path
import fnmatch

import cartopy
import geopandas as gpd
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from cartopy.io import shapereader as shpreader
from matplotlib import pyplot as plt
from tbparse import SummaryReader

import definitions
from hydromtl.utils.hydro_utils import unserialize_json
from hydromtl.data.data_dict import data_sources_dict
from hydromtl.models.trainer import (
    load_result,
    save_result,
    train_and_evaluate,
)
from hydromtl.utils.hydro_stat import stat_error, ecdf
from hydromtl.data.config import default_config_file, update_cfg, cmd
from hydromtl.visual.plot_stat import plot_ecdfs_matplot, plot_ts, plot_rainfall_runoff
from hydromtl.utils import hydro_constant, hydro_utils
from hydromtl.data.source.data_camels import Camels
from hydromtl.data.source.data_constant import (
    ET_MODIS_NAME,
    PRCP_ERA5LAND_NAME,
)
from scripts.app_constant import VAR_C_CHOSEN_FROM_GAGES_II


def get_json_file(cfg_dir):
    json_files_lst = []
    json_files_ctime = []
    for file in os.listdir(cfg_dir):
        if (
            fnmatch.fnmatch(file, "*.json")
            and "_stat" not in file  # statistics json file
            and "_dict" not in file  # data cache json file
        ):
            json_files_lst.append(os.path.join(cfg_dir, file))
            json_files_ctime.append(os.path.getctime(os.path.join(cfg_dir, file)))
    sort_idx = np.argsort(json_files_ctime)
    cfg_file = json_files_lst[sort_idx[-1]]
    cfg_json = unserialize_json(cfg_file)
    if cfg_json["data_params"]["test_path"] != cfg_dir:
        # sometimes we will use files copied from other device, so the dir is not correct for this device
        update_cfg_as_move_to_another_pc(cfg_json=cfg_json)

    return cfg_json


def update_cfg_as_move_to_another_pc(cfg_json):
    """update cfg as move to another pc

    Returns
    -------
    _type_
        _description_
    """
    cfg_json["data_params"]["test_path"] = get_the_new_path_with_diff_part(
        cfg_json, "test_path"
    )
    cfg_json["data_params"]["data_path"] = get_the_new_path_with_diff_part(
        cfg_json, "data_path"
    )
    cfg_json["data_params"]["cache_path"] = get_the_new_path_with_diff_part(
        cfg_json, "cache_path"
    )
    cfg_json["data_params"]["validation_path"] = get_the_new_path_with_diff_part(
        cfg_json, "validation_path"
    )


def get_the_new_path_with_diff_part(cfg_json, replace_item):
    the_item = cfg_json["data_params"][replace_item]
    if the_item is None:
        return None
    common_path_name = get_common_path_name(the_item, replace_item)
    dff_path_name = (
        definitions.DATASET_DIR if replace_item == "data_path" else definitions.ROOT_DIR
    )
    if type(common_path_name) is list:
        return [os.path.join(dff_path_name, a_path) for a_path in common_path_name]
    return os.path.join(dff_path_name, common_path_name)


def get_common_path_name(origin_pc_test_path, replace_item):
    if type(origin_pc_test_path) is list:
        return [
            get_common_path_name(a_path, replace_item) for a_path in origin_pc_test_path
        ]
    if origin_pc_test_path.startswith("/"):
        # linux
        origin_pc_test_path_lst = origin_pc_test_path.split("/")
    else:
        # windows
        origin_pc_test_path_lst = origin_pc_test_path.split("\\")
    # NOTE: this is a hard code
    if replace_item == "data_path":
        pos_lst = [i for i, e in enumerate(origin_pc_test_path_lst) if e == "data"]
        the_root_dir = definitions.DATASET_DIR
    else:
        pos_lst = [i for i, e in enumerate(origin_pc_test_path_lst) if e == "HydroSPB"]
        the_root_dir = definitions.ROOT_DIR
    if not pos_lst:
        raise ValueError("Can not find the common path name")
    elif len(pos_lst) == 1:
        where_start_same_in_origin_pc = pos_lst[0] + 1
    else:
        for i in pos_lst:
            if os.path.exists(
                os.path.join(
                    the_root_dir, os.sep.join(origin_pc_test_path_lst[i + 1 :])
                )
            ):
                where_start_same_in_origin_pc = i + 1
                break

    return os.sep.join(origin_pc_test_path_lst[where_start_same_in_origin_pc:])


def get_latest_file_in_a_lst(lst):
    """get the latest file in a list

    Parameters
    ----------
    lst : list
        list of files

    Returns
    -------
    str
        the latest file
    """
    lst_ctime = [os.path.getctime(file) for file in lst]
    sort_idx = np.argsort(lst_ctime)
    return lst[sort_idx[-1]]


def get_lastest_weight_path(weight_path_dir):
    """Get the last modified weight file

    Parameters
    ----------
    weight_path_dir : _type_
        _description_

    Returns
    -------
    str
        the path of the weight file
    """
    pth_files_lst = [
        os.path.join(weight_path_dir, file)
        for file in os.listdir(weight_path_dir)
        if fnmatch.fnmatch(file, "*.pth")
    ]
    return get_latest_file_in_a_lst(pth_files_lst)


def evaluate_a_model(
    exp,
    example="camels",
    epoch=None,
    train_period=None,
    test_period=None,
    save_result_name=None,
    is_tl=False,
    sub_exp=None,
    data_dir=None,
    device=None,
    dpl_param=None,
):
    """
    Evaluate a trained model

    Parameters
    ----------
    exp
        the name of exp, such as "exp511"
    example
        first sub-dir in "example" directory: "camels", "gages", ... default is the former
    epoch
        model saved in which epoch is used here
    train_period
        the period of training data for model
    test_period
        the period of testing data for model
    save_result_name
        the name of the result file, default is None
    sub_exp
        the name of sub exp, default is None,
        which is for saved models during training for different hyper-parameters settings
    data_dir
        the directory of data source, default is None
        when move the trained model from one machine to another, this will be useful

    Returns
    -------
    None
    """
    cfg_dir_flow = os.path.join(
        definitions.ROOT_DIR, "hydroSPB", "example", example, exp
    )
    cfg_flow = get_json_file(cfg_dir_flow)
    if data_dir is not None:
        cfg_flow["data_params"]["data_path"] = data_dir
        cfg_flow["data_params"]["test_path"] = cfg_dir_flow
        cfg_flow["data_params"]["cache_path"] = cfg_dir_flow
        cfg_flow["data_params"]["validation_path"] = cfg_dir_flow
    cfg_flow["model_params"]["continue_train"] = False
    if train_period is not None:
        cfg_flow["data_params"]["t_range_train"] = train_period
        cfg_flow["data_params"]["cache_read"] = False
        # don't save the cache file,
        # because we will evaluate the model with different settings in the same directory
        cfg_flow["data_params"]["cache_write"] = False
    if test_period is not None:
        cfg_flow["data_params"]["t_range_test"] = test_period
        cfg_flow["data_params"]["cache_read"] = False
        cfg_flow["data_params"]["cache_write"] = False
    if epoch is None:
        epoch = cfg_flow["evaluate_params"]["test_epoch"]
    else:
        # the epoch is used for test
        cfg_flow["evaluate_params"]["test_epoch"] = epoch
    train_epoch = cfg_flow["training_params"]["epochs"]
    if epoch != train_epoch:
        cfg_flow["training_params"]["epochs"] = epoch
    if device is not None:
        cfg_flow["training_params"]["device"] = device
    if sub_exp is not None:
        weight_path = os.path.join(cfg_dir_flow, sub_exp, f"model_Ep{str(epoch)}.pth")
    else:
        weight_path = os.path.join(cfg_dir_flow, f"model_Ep{str(epoch)}.pth")
    if not os.path.isfile(weight_path):
        weight_path = os.path.join(cfg_dir_flow, f"model_Ep{str(epoch)}.pt")
    cfg_flow["model_params"]["weight_path"] = weight_path
    if cfg_flow["data_params"]["cache_read"]:
        cfg_flow["data_params"]["cache_write"] = False
    if is_tl:
        # we evaluate a tl model, so we need to set tl_tag to False to avoid it perform tl modeling again
        cfg_flow["model_params"]["model_param"]["tl_tag"] = False
    if dpl_param is not None:
        cfg_flow["model_params"]["model_param"].update(dpl_param)
    train_and_evaluate(cfg_flow)
    # new name for results, becuase we will evaluate the model with different settings in the same directory
    pred, obs = load_result(
        cfg_flow["data_params"]["test_path"], epoch, not_only_1out=True
    )
    if save_result_name is None:
        pred_name = "flow_pred"
        obs_name = "flow_obs"
    else:
        pred_name = save_result_name + "_pred"
        obs_name = save_result_name + "_obs"
    save_result(
        cfg_flow["data_params"]["test_path"],
        epoch,
        pred,
        obs,
        pred_name=pred_name,
        obs_name=obs_name,
    )
    print("Call a trained model and save its evaluation results")


def predict_in_test_period_with_model(new_exp_args, cache_cfg_dir, weight_path):
    """Prediction in a test period with the given trained model for a new experiment

    Parameters
    ----------
    new_exp_args : str
        arguments for new experiment
    cache_cfg_dir : str
        the directory of cache file
    weight_path : str
        the path of trained model's weight file
    """
    cfg = default_config_file()
    update_cfg(cfg, new_exp_args)
    if cache_cfg_dir is not None:
        # test_data_dict.json is a flag for cache existing,
        # if exsits, we don't need to write cache file again
        cfg["data_params"]["cache_write"] = not os.path.exists(
            os.path.join(cache_cfg_dir, "test_data_dict.json")
        )
        cfg["data_params"]["cache_path"] = cache_cfg_dir
    cfg["data_params"]["cache_read"] = True
    cfg["model_params"]["continue_train"] = False
    cfg["model_params"]["weight_path"] = weight_path
    if weight_path is None:
        cfg["model_params"]["continue_train"] = True
    train_and_evaluate(cfg)
    print("Call a trained model and test it in a new period")


def plot_ecdf_func(
    inds_all_lst,
    cases_exps_legends_together,
    save_path,
    dash_lines=None,
    ecdf_fig_size=(6, 4),
    colors="rbkgcmy",
    x_str="NSE",
    x_interval=0.1,
    x_lim=(0, 1),
    show_legend=True,
    legend_font_size=16,
):
    """
    Plot ECDF figs for a list of NSE arrays

    Parameters
    ----------
    inds_all_lst
        list of NSE arrays
    cases_exps_legends_together
        exps' names for the legend
    save_path
        where we save the fig
    dash_lines
        if the line will be a dash line
    ecdf_fig_size
        fig's size
    colors
        colors for each line
    x_str
        the name of x-axis
    x_interval
        interval of x-axis
    x_lim
        limits of x-axis
    show_legend
        if True, show legend

    Returns
    -------
    plt.figure
        the plot
    """
    if dash_lines is None:
        dash_lines = [True, False, False, False, False, False]
    print("plot CDF")
    xs1 = []
    ys1 = []
    for ind_ in inds_all_lst:
        xi, yi = ecdf(np.nan_to_num(ind_))
        xs1.append(xi)
        ys1.append(yi)
    fig, ax = plot_ecdfs_matplot(
        xs1,
        ys1,
        cases_exps_legends_together,
        dash_lines=dash_lines,
        x_str=x_str,
        y_str="CDF",
        colors=colors,
        fig_size=ecdf_fig_size,
        x_interval=x_interval,
        x_lim=x_lim,
        show_legend=show_legend,
        legend_font_size=legend_font_size,
    )
    plt.tight_layout()
    FIGURE_DPI = 600
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def predict_new_gages_exp(
    exp,
    gage_id_file,
    weight_path,
    continue_train,
    random_seed,
    cache_path=None,
    gages_id=None,
    stat_dict_file=None,
):
    project_name = "gages/" + exp
    args = cmd(
        sub=project_name,
        source_path=os.path.join(definitions.DATASET_DIR, "gages"),
        source="GAGES",
        download=0,
        ctx=[0],
        model_name="KuaiLSTM",
        model_param={
            "n_input_features": 37,
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        weight_path=weight_path,
        continue_train=continue_train,
        opt="Adadelta",
        loss_func="RMSESum",
        rs=random_seed,
        train_period=["1990-01-01", "2000-01-01"],
        test_period=["2000-01-01", "2010-01-01"],
        cache_write=1,
        cache_read=1,
        scaler="DapengScaler",
        data_loader="StreamflowDataModel",
        train_epoch=300,
        te=300,
        save_epoch=50,
        batch_size=100,
        rho=365,
        var_c=VAR_C_CHOSEN_FROM_GAGES_II,
        gage_id_file=gage_id_file,
        gage_id=gages_id,
        stat_dict_file=stat_dict_file,
    )
    predict_in_test_period_with_model(
        args, weight_path=weight_path, cache_cfg_dir=cache_path
    )


def read_dl_models_q_for_1basin1fold(
    exp, epoch, cv_fold_i, streamflow_unit="ft3/s", **kwargs
):
    """read dl models simulations for one basin for one fold

    Parameters
    ----------
    exp : _type_
        _description_
    epoch : _type_
        _description_
    cv_fold_i : _type_
        _description_
    streamflow_unit : str, optional
        the unit of streamflow of the saved result, we will convert it to m3/s, by default "ft3/s"
    kwargs : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    the_exp = exp + "0" + str(cv_fold_i)
    cfg_dir_flow = os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", the_exp)
    cfg_flow = get_json_file(cfg_dir_flow)
    save_result_train_name = f"fold{str(cv_fold_i)}train"
    save_result_valid_name = f"fold{str(cv_fold_i)}valid"
    pred_train_name = save_result_train_name + "_pred"
    pred_valid_name = save_result_valid_name + "_pred"
    obs_train_name = save_result_train_name + "_obs"
    obs_valid_name = save_result_valid_name + "_obs"
    pred_train, obs_train = load_result(
        cfg_flow["data_params"]["test_path"], epoch, pred_train_name, obs_train_name
    )
    pred_valid, obs_valid = load_result(
        cfg_flow["data_params"]["test_path"], epoch, pred_valid_name, obs_valid_name
    )
    pred_train_ = hydro_constant.convert_unit(
        pred_train,
        streamflow_unit,
        hydro_constant.unified_default_unit["streamflow"],
        **kwargs,
    )
    pred_valid_ = hydro_constant.convert_unit(
        pred_valid,
        streamflow_unit,
        hydro_constant.unified_default_unit["streamflow"],
        **kwargs,
    )
    obs_train_ = hydro_constant.convert_unit(
        obs_train,
        streamflow_unit,
        hydro_constant.unified_default_unit["streamflow"],
        **kwargs,
    )
    obs_valid_ = hydro_constant.convert_unit(
        obs_valid,
        streamflow_unit,
        hydro_constant.unified_default_unit["streamflow"],
        **kwargs,
    )
    return pred_train_, obs_train_, pred_valid_, obs_valid_


def read_dl_models_q_for_1basin(
    exp, epoch, cv_fold=2, streamflow_unit="ft3/s", **kwargs
):
    """read dl models streamflow for one basin

    Returns
    -------
    streamflow_unit
        the unit of streamflow of the saved result, we will convert it to m3/s
    """
    pred_trains_lst = []
    pred_valids_lst = []
    obs_trains_lst = []
    obs_valids_lst = []
    for i in range(cv_fold):
        (
            pred_train_,
            obs_train_,
            pred_valid_,
            obs_valid_,
        ) = read_dl_models_q_for_1basin1fold(exp, epoch, i, streamflow_unit, **kwargs)
        the_exp = exp + "0" + str(i)
        cfg_dir_flow = os.path.join(
            definitions.ROOT_DIR, "hydroSPB", "example", the_exp
        )
        cfg_flow = get_json_file(cfg_dir_flow)
        warmup = cfg_flow["data_params"]["warmup_length"]
        pred_train = pd.DataFrame(
            pred_train_.T,
            index=hydro_utils.t_range_days(cfg_flow["data_params"]["t_range_train"])[
                warmup:
            ],
            columns=cfg_flow["data_params"]["object_ids"],
        )
        pred_valid = pd.DataFrame(
            pred_valid_.T,
            index=hydro_utils.t_range_days(cfg_flow["data_params"]["t_range_valid"])[
                warmup:
            ],
            columns=cfg_flow["data_params"]["object_ids"],
        )
        obs_train = pd.DataFrame(
            obs_train_.T,
            index=hydro_utils.t_range_days(cfg_flow["data_params"]["t_range_train"])[
                warmup:
            ],
            columns=cfg_flow["data_params"]["object_ids"],
        )
        obs_valid = pd.DataFrame(
            obs_valid_.T,
            index=hydro_utils.t_range_days(cfg_flow["data_params"]["t_range_valid"])[
                warmup:
            ],
            columns=cfg_flow["data_params"]["object_ids"],
        )
        pred_trains_lst.append(pred_train)
        pred_valids_lst.append(pred_valid)
        obs_trains_lst.append(obs_train)
        obs_valids_lst.append(obs_valid)
    return pred_trains_lst, pred_valids_lst, obs_trains_lst, obs_valids_lst


def read_dl_models_q_metric_for_1basin(
    exp, epoch, cv_fold=2, streamflow_unit="ft3/s", **kwargs
):
    """read the metrics of DL models for one basin in k-fold cross validation

    Parameters
    ----------
    epoch : int
        the epoch of the DL model
    cv_fold : int, optional
        the number of folds in cross validation, by default 2
    """
    inds_df_trains_lst = []
    inds_df_valids_lst = []
    for i in range(cv_fold):
        (
            pred_train_,
            obs_train_,
            pred_valid_,
            obs_valid_,
        ) = read_dl_models_q_for_1basin1fold(exp, epoch, i, streamflow_unit, **kwargs)
        inds_df_train = pd.DataFrame(stat_error(obs_train_, pred_train_))
        inds_df_valid = pd.DataFrame(stat_error(obs_valid_, pred_valid_))
        inds_df_trains_lst.append(inds_df_train)
        inds_df_valids_lst.append(inds_df_valid)
    inds_df_trains = pd.concat(inds_df_trains_lst).mean()
    inds_df_valids = pd.concat(inds_df_valids_lst).mean()
    return inds_df_trains, inds_df_valids


def read_dl_models_et_for_1basin1fold(exp, epoch, cv_fold_i, et_type=ET_MODIS_NAME):
    the_exp = exp + "0" + str(cv_fold_i)
    exp_dir = os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", the_exp)
    cfg_exp = get_json_file(exp_dir)
    gage_id_lst = cfg_exp["data_params"]["object_ids"]
    train_period = cfg_exp["data_params"]["t_range_train"]
    test_period = cfg_exp["data_params"]["t_range_test"]
    train_period_lst = hydro_utils.t_range_days(train_period)
    test_period_lst = hydro_utils.t_range_days(test_period)
    warmup = cfg_exp["data_params"]["warmup_length"]
    etobs_train = read_et_obs_for_1basin(gage_id_lst[0], train_period, et_type=et_type)
    etobs_train = etobs_train.reshape(etobs_train.shape[0], etobs_train.shape[1]).T
    etobs_test = read_et_obs_for_1basin(gage_id_lst[0], test_period, et_type=et_type)
    etobs_test = etobs_test.reshape(etobs_test.shape[0], etobs_test.shape[1]).T
    etsim_train = np.load(
        os.path.join(exp_dir, f"epoch{str(epoch)}fold{str(cv_fold_i)}train_pred.npy")
    )[:, :, -1].T

    etsim_test = np.load(
        os.path.join(exp_dir, f"epoch{str(epoch)}fold{str(cv_fold_i)}valid_pred.npy")
    )[:, :, -1].T

    etsim_train_result = pd.DataFrame(
        etsim_train,
        index=pd.to_datetime(train_period_lst).values[warmup:].astype("datetime64[D]"),
        columns=gage_id_lst,
    )
    etsim_test_result = pd.DataFrame(
        etsim_test,
        index=pd.to_datetime(test_period_lst).values[warmup:].astype("datetime64[D]"),
        columns=gage_id_lst,
    )
    etobs_train_result = pd.DataFrame(
        etobs_train[warmup:],
        index=pd.to_datetime(train_period_lst).values[warmup:].astype("datetime64[D]"),
        columns=gage_id_lst,
    )
    etobs_test_result = pd.DataFrame(
        etobs_test[warmup:],
        index=pd.to_datetime(test_period_lst).values[warmup:].astype("datetime64[D]"),
        columns=gage_id_lst,
    )
    return (
        etsim_train_result,
        etsim_test_result,
        etobs_train_result,
        etobs_test_result,
    )


def read_dl_models_et_for_1basin(exp, epoch, cv_fold, et_type=ET_MODIS_NAME):
    etsim_train_result_lst = []
    etobs_train_result_lst = []
    etsim_test_result_lst = []
    etobs_test_result_lst = []
    for cv_fold_i in range(cv_fold):
        (
            etsim_train_result,
            etsim_test_result,
            etobs_train_result,
            etobs_test_result,
        ) = read_dl_models_et_for_1basin1fold(exp, epoch, cv_fold_i, et_type=et_type)
        etsim_train_result_lst.append(etsim_train_result)
        etobs_train_result_lst.append(etobs_train_result)
        etsim_test_result_lst.append(etsim_test_result)
        etobs_test_result_lst.append(etobs_test_result)
    return (
        etsim_train_result_lst,
        etsim_test_result_lst,
        etobs_train_result_lst,
        etobs_test_result_lst,
    )


def read_dl_models_et_metric_for_1basin(exp, epoch, cv_fold=2, **kwargs):
    """read the metrics of DL models for one basin in k-fold cross validation

    Parameters
    ----------
    epoch : int
        the epoch of the DL model
    cv_fold : int, optional
        the number of folds in cross validation, by default 2
    """
    inds_df_trains_lst = []
    inds_df_valids_lst = []
    for i in range(cv_fold):
        (
            pred_train_,
            pred_valid_,
            obs_train_,
            obs_valid_,
        ) = read_dl_models_et_for_1basin1fold(exp, epoch, i, **kwargs)
        inds_df_train = pd.DataFrame(
            stat_error(obs_train_.values.T, pred_train_.values.T, fill_nan="mean")
        )
        inds_df_valid = pd.DataFrame(
            stat_error(obs_valid_.values.T, pred_valid_.values.T, fill_nan="mean")
        )
        inds_df_trains_lst.append(inds_df_train)
        inds_df_valids_lst.append(inds_df_valid)
    inds_df_trains = pd.concat(inds_df_trains_lst).mean()
    inds_df_valids = pd.concat(inds_df_valids_lst).mean()
    return inds_df_trains, inds_df_valids


def read_tb_log(
    a_exp, best_batchsize, exp_example="gages", where_save="transfer_learning"
):
    """Copy a recent log file to the current directory and read the log file.

    Parameters
    ----------
    a_exp : _type_
        _description_
    best_batchsize : _type_
        _description_
    exp_example : str, optional
        _description_, by default "gages"
    where_save : str, optional
        A directory in "app" directory, by default "transfer_learning"

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    FileNotFoundError
        _description_
    """
    log_dir = os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        exp_example,
        a_exp,
        f"opt_Adadelta_lr_1.0_bsize_{str(best_batchsize)}",
    )
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log dir {log_dir} not found!")
    result_dir = os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "app",
        where_save,
        "results",
        "tensorboard",
        a_exp,
        f"opt_Adadelta_lr_1.0_bsize_{str(best_batchsize)}",
    )
    copy_latest_tblog_file(log_dir, result_dir)
    scalar_file = os.path.join(result_dir, "scalars.csv")
    if not os.path.exists(scalar_file):
        reader = SummaryReader(result_dir)
        df_scalar = reader.scalars
        df_scalar.to_csv(scalar_file, index=False)
    else:
        df_scalar = pd.read_csv(scalar_file)

    # reader = SummaryReader(result_dir)
    histgram_file = os.path.join(result_dir, "histograms.pkl")
    if not os.path.exists(histgram_file):
        reader = SummaryReader(result_dir, pivot=True)
        df_histgram = reader.histograms
        # https://www.statology.org/pandas-save-dataframe/
        df_histgram.to_pickle(histgram_file)
    else:
        df_histgram = pd.read_pickle(histgram_file)
    return df_scalar, df_histgram


def get_latest_event_file(event_file_lst):
    """Get the latest event file in the current directory.

    Returns
    -------
    str
        The latest event file.
    """
    event_files = [Path(f) for f in event_file_lst]
    event_file_names_lst = [event_file.stem.split(".") for event_file in event_files]
    ctimes = [
        int(event_file_names[event_file_names.index("tfevents") + 1])
        for event_file_names in event_file_names_lst
    ]
    return event_files[ctimes.index(max(ctimes))]


def get_latest_pbm_param_file(param_dir):
    """Get the latest parameter file of physics-based models in the current directory.

    Parameters
    ----------
    param_dir : str
        The directory of parameter files.

    Returns
    -------
    str
        The latest parameter file.
    """
    param_file_lst = [
        os.path.join(param_dir, f)
        for f in os.listdir(param_dir)
        if f.startswith("pb_params") and f.endswith(".csv")
    ]
    param_files = [Path(f) for f in param_file_lst]
    param_file_names_lst = [param_file.stem.split("_") for param_file in param_files]
    ctimes = [
        int(param_file_names[param_file_names.index("params") + 1])
        for param_file_names in param_file_names_lst
    ]
    return param_files[ctimes.index(max(ctimes))]


# Prepare temp dirs for storing event files
def copy_latest_tblog_file(log_dir, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        copy_lst = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.startswith("events")
        ]
        copy_file = get_latest_event_file(copy_lst)
        shutil.copy(copy_file, result_dir)


def plot_ts_for_basin_fold(
    leg_lst,
    basin_id,
    fold,
    step_lst,
    value_lst,
    ylabel,
    where_save="transfer_learning",
    sub_dir=os.path.join("results", "tensorboard"),
    batch_size=None,
):
    """Lineplot for loss and metric of DL models for one basin in a fold experiment

    Parameters
    ----------
    leg_lst : list
        a list of legends
    basin_id : _type_
        _description_
    fold : _type_
        _description_
    step_lst : _type_
        _description_
    value_lst : _type_
        _description_
    ylabel : _type_
        _description_
    where_save : str, optional
        _description_, by default "transfer_learning"
    """
    result_dir = os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "app",
        where_save,
        sub_dir,
    )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plot_ts(
        step_lst,
        value_lst,
        leg_lst=leg_lst,
        fig_size=(6, 4),
        xlabel="代数",
        ylabel=ylabel,
    )
    if batch_size is None:
        plt.savefig(
            os.path.join(
                result_dir,
                f"{basin_id}_fold{fold}_{ylabel}.png",
            ),
            dpi=600,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            os.path.join(
                result_dir,
                f"{basin_id}_fold{fold}_{ylabel}_bsize{batch_size}.png",
            ),
            dpi=600,
            bbox_inches="tight",
        )
