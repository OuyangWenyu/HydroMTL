"""
Author: Wenyu Ouyang
Date: 2022-07-23 10:51:52
LastEditTime: 2023-04-06 22:01:58
LastEditors: Wenyu Ouyang
Description: Plots utils for MTL results
FilePath: /HydroMTL/hydromtl/scripts/mtl_results_utils.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydromtl.scripts.streamflow_utils import (
    get_json_file,
    get_lastest_weight_path,
    predict_in_test_period_with_model,
)
from hydromtl.data.config import cmd, default_config_file, update_cfg
from hydromtl.data.source.data_camels import Camels
from hydromtl.models.trainer import stat_result, train_and_evaluate
from hydromtl.utils.hydro_stat import stat_error
from hydromtl.visual.plot_stat import (
    plot_boxes_matplotlib,
    plot_boxs,
    plot_ts,
    plot_map_carto,
)
from hydromtl.data.source.data_constant import (
    PRCP_DAYMET_NAME,
    PRCP_NLDAS_NAME,
    Q_CAMELS_US_NAME,
    ET_MODIS_NAME,
    SSM_SMAP_NAME,
    Q_CAMELS_CC_NAME,
    PET_NLDAS_NAME,
    PET_MODIS_NAME,
    NLDAS_NAME,
    ERA5LAND_NAME,
)
from hydromtl.scripts.app_constant import (
    VAR_C_CHOSEN_FROM_CAMELS_US,
    VAR_T_CHOSEN_FROM_NLDAS,
    VAR_C_CHOSEN_FROM_CAMELS_CC,
    VAR_T_CHOSEN_FROM_ERA5LAND,
)
from hydromtl.utils import hydro_constant


def plot_multi_single_comp_flow_boxes(
    inds_all_lst,
    cases_exps_legends_together,
    save_path,
    rotation=0,
    x_name="λ",
    y_name="NSE",
):
    """plot boxes with seaborn

    Parameters
    ----------
    inds_all_lst : _type_
        _description_
    cases_exps_legends_together : _type_
        _description_
    save_path : _type_
        _description_
    rotation : int, optional
        the rotation angle of x-labels, by default 0
    x_name
        the x-label, by default λ
    y_name
        the x-label, by default NSE
    """
    frames = []
    for i in range(len(cases_exps_legends_together)):
        inds_test = inds_all_lst[i]
        # TODO: see what happened here, the interface is not unified
        if type(inds_test) == list:
            inds_test = np.array(inds_test)
        df_dict_i = {}
        str_i = cases_exps_legends_together[i]
        df_dict_i[x_name] = np.full([inds_test.size], str_i)
        df_dict_i[y_name] = inds_test
        df_i = pd.DataFrame(df_dict_i)
        frames.append(df_i)
    inds_box = pd.concat(frames)
    plot_boxs(inds_box, x_name, y_name, rotation=rotation, show_median=True)
    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
    )


def read_multi_single_exps_results(
    exps_lst,
    best_valid_idx=None,
    var_idx=0,
    metric="NSE",
    flow_unit="ft3/s",
    et_unit="0.1mm/day",
    single_is_flow=True,
    flow_idx_in_mtl=0,
    et_idx_in_mtl=1,
    ensemble=0,
    return_value=False,
):
    """
    Read results for one variable's prediction in exps_lst

    Parameters
    ----------
    exps_lst : list
        [single_output_exp, multi_output_exp1, multi_output_exp2, ...]
    best_valid_idx
        if it is not None, chose best according to it
    var_idx:int
        the variable's index in multiple variables
    metric:str
        the metric we will use, default is NSE
    flow_unit:str
        unit of streamflow, default is ft3/s
    single_is_flow:bool
        the single task is streamflow, default is true
    flow_idx_in_mtl
        the idx of streamflow in mtl, default is 0
    ensemble
        if it is 1, calculate ensemble mean,
        elif 0 just calculate mean of metric, by default 0
        else don't include mean
    return_value
        if it is true, return the pred and obs values, by default False
    Returns
    -------
    list
        list of all single task and multiple task learning models' results,
        and the final two are ensemble mean and best of all mtl models' multiple arrays
        if there are not only one MTL model's result
    """
    inds_all_lst = []
    preds_all_lst = []
    obss_all_lst = []
    cfg_dir_var = os.path.join(
        definitions.ROOT_DIR, "hydroSPB", "example", "camels", exps_lst[0]
    )
    cfg_var = get_json_file(cfg_dir_var)
    var_name = (
        hydro_constant.streamflow.name
        if single_is_flow
        else hydro_constant.evapotranspiration.name
    )
    inds_df, pred_stl, obs_stl = stat_result(
        cfg_var["data_params"]["test_path"],
        cfg_var["evaluate_params"]["test_epoch"],
        fill_nan=cfg_var["evaluate_params"]["fill_nan"],
        unit=flow_unit if single_is_flow else et_unit,
        return_value=True,
        var_name=var_name,
    )
    inds_all_lst.append(inds_df[metric].values)
    preds_all_lst.append(pred_stl)
    # all obs are the same, so just append this one
    obss_all_lst.append(obs_stl)
    units = [None] * 2
    var_names = [hydro_constant.streamflow.name, hydro_constant.evapotranspiration.name]
    units[flow_idx_in_mtl] = flow_unit
    units[et_idx_in_mtl] = et_unit
    preds = []
    obss = []
    for i in range(1, len(exps_lst)):
        cfg_dir_flow_other = os.path.join(
            definitions.ROOT_DIR, "hydroSPB", "example", "camels", exps_lst[i]
        )
        cfg_flow_other = get_json_file(cfg_dir_flow_other)
        inds_df1, pred, obs = stat_result(
            cfg_flow_other["data_params"]["test_path"],
            cfg_flow_other["evaluate_params"]["test_epoch"],
            fill_nan=cfg_flow_other["evaluate_params"]["fill_nan"],
            unit=units,
            return_value=True,
            var_name=var_names,
        )
        inds_all_lst.append(inds_df1[var_idx][metric].values)
        preds.append(pred[var_idx])
        obss.append(obs[var_idx])
    preds_all_lst = preds_all_lst + preds
    if len(exps_lst) > 2:
        if ensemble == 1:
            pred_ensemble = np.array(preds).mean(axis=0)
            obs_ensemble = np.array(obss).mean(axis=0)
            inds_ensemble = stat_error(
                obs_ensemble,
                pred_ensemble,
                fill_nan=cfg_flow_other["evaluate_params"]["fill_nan"][var_idx],
            )
            mtl_mean_results = pd.DataFrame(inds_ensemble)[metric]
        elif ensemble == 0:
            # the first element is stl model, so index start from 1
            mtl_mean_results = np.array(inds_all_lst[1:]).mean(axis=0)
        else:
            mtl_mean_results = None
        if best_valid_idx is None:
            mtl_best_results = np.array(inds_all_lst[1:]).max(axis=0)
            mtl_best_results_where = np.array(inds_all_lst[1:]).argmax(axis=0)
            pred_best = np.array(
                [
                    np.array(preds)[idx, i, :]
                    for i, idx in enumerate(mtl_best_results_where)
                ]
            )
        else:
            mtl_results = np.array(inds_all_lst[1:])
            mtl_best_results = np.array(
                [mtl_results[idx, i] for i, idx in enumerate(best_valid_idx)]
            )
            pred_best = np.array(
                [np.array(preds)[idx, i, :] for i, idx in enumerate(best_valid_idx)]
            )
            mtl_best_results_where = np.array(inds_all_lst[1:]).argmax(axis=0)
        if mtl_mean_results is not None:
            # mean results are the second last one
            inds_all_lst.append(mtl_mean_results)
            if ensemble == 1:
                preds_all_lst.append(pred_ensemble)
        # best results are the last one
        inds_all_lst.append(mtl_best_results)
        preds_all_lst.append(pred_best)
        if return_value:
            return inds_all_lst, mtl_best_results_where, preds_all_lst, obss_all_lst
        return inds_all_lst, mtl_best_results_where
    if return_value:
        return inds_all_lst, preds_all_lst, obss_all_lst
    return inds_all_lst


def predict_new_et_exp(
    exp,
    weight_path,
    train_period,
    test_period,
    cache_path=None,
    gage_id_file=None,
    stat_dict_file=None,
    scaler_params=None,
):
    """use a trained model to predict ET for new basins

    Parameters
    ----------
    exp : _type_
        _description_
    weight_path : _type_
        _description_
    train_period : _type_
        _description_
    test_period : _type_
        _description_
    cache_path : _type_, optional
        _description_, by default None
    gage_id_file : _type_, optional
        _description_, by default None
    stat_dict_file:
        statistics file from trained models,
    """
    project_name = "camels/" + exp
    if gage_id_file is None:
        gage_id = "ALL"
    else:
        gage_id = pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values.tolist()
    if scaler_params is None:
        # same with config.py file
        scaler_params = {
            "basin_norm_cols": [
                Q_CAMELS_US_NAME,
                "streamflow",
                Q_CAMELS_CC_NAME,
                "qobs",
            ],
            "gamma_norm_cols": [
                PRCP_DAYMET_NAME,
                "pr",
                # PRCP_ERA5LAND_NAME is same as PRCP_NLDAS_NAME
                PRCP_NLDAS_NAME,
                "pre",
                # pet may be negative, but we set negative as 0 because of gamma_norm_cols
                # https://earthscience.stackexchange.com/questions/12031/does-negative-reference-evapotranspiration-make-sense-using-fao-penman-monteith
                "pet",
                # PET_ERA5LAND_NAME is same as PET_NLDAS_NAME
                PET_NLDAS_NAME,
                ET_MODIS_NAME,
                "LE",
                PET_MODIS_NAME,
                "PLE",
                "GPP",
                "Ec",
                "Es",
                "Ei",
                "ET_water",
                "ET_sum",
                SSM_SMAP_NAME,
                "susm",
                "smp",
                "ssma",
                "susma",
            ],
        }
    args = cmd(
        sub=project_name,
        source_path=[
            os.path.join(definitions.DATASET_DIR, "camelsflowet"),
            os.path.join(definitions.DATASET_DIR, "modiset4camels"),
            os.path.join(definitions.DATASET_DIR, "camels", "camels_us"),
            os.path.join(definitions.DATASET_DIR, "nldas4camels"),
            os.path.join(definitions.DATASET_DIR, "smap4camels"),
        ],
        source="CAMELS_FLOW_ET",
        download=0,
        ctx=[0],
        model_name="KuaiLSTM",
        model_param={
            "n_input_features": 23,
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [2],
            "item_weight": [1.0],
            "device": [0],
        },
        cache_write=1,
        batch_size=100,
        rho=365,
        opt="Adadelta",
        rs=1234,
        var_t=VAR_T_CHOSEN_FROM_NLDAS,
        var_t_type=[NLDAS_NAME],
        var_out=[ET_MODIS_NAME],
        train_period=train_period,
        test_period=test_period,
        data_loader="StreamflowDataModel",
        scaler="DapengScaler",
        scaler_params=scaler_params,
        n_output=1,
        train_epoch=300,
        te=300,
        save_epoch=10,
        fill_nan=["mean"],
        gage_id=gage_id,
        stat_dict_file=stat_dict_file,
    )
    predict_in_test_period_with_model(
        args, weight_path=weight_path, cache_cfg_dir=cache_path
    )


def predict_new_q_exp(
    exp,
    weight_path,
    train_period,
    test_period,
    cache_path=None,
    gage_id_file=None,
    stat_dict_file=None,
):
    project_name = "camels/" + exp
    if gage_id_file is None:
        gage_id = "ALL"
    else:
        gage_id = pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values.tolist()
    args = cmd(
        sub=project_name,
        source_path=[
            os.path.join(definitions.DATASET_DIR, "nldas4camels"),
            os.path.join(definitions.DATASET_DIR, "camels", "camels_us"),
        ],
        source="NLDAS_CAMELS",
        download=0,
        ctx=[0],
        model_name="KuaiLSTM",
        model_param={
            "n_input_features": 23,
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        opt="Adadelta",
        loss_func="RMSESum",
        rs=1234,
        cache_write=1,
        batch_size=100,
        rho=365,
        var_t=VAR_T_CHOSEN_FROM_NLDAS,
        var_t_type=[NLDAS_NAME],
        train_period=train_period,
        test_period=test_period,
        data_loader="StreamflowDataModel",
        scaler="DapengScaler",
        train_epoch=300,
        te=300,
        save_epoch=20,
        gage_id=gage_id,
        stat_dict_file=stat_dict_file,
    )
    predict_in_test_period_with_model(
        args, weight_path=weight_path, cache_cfg_dir=cache_path
    )


def predict_new_mtl_exp(
    exp,
    targets,
    loss_weights,
    weight_path,
    train_period,
    test_period,
    cache_path=None,
    gage_id_file=None,
    stat_dict_file=None,
    scaler_params=None,
    loss_func="MultiOutLoss",
    alpah=None,
):
    project_name = "camels/" + exp
    if targets == [Q_CAMELS_US_NAME, ET_MODIS_NAME]:
        data_gap = [0, 2]
        fill_nan = ["no", "mean"]
        n_output = 2
    elif targets == [Q_CAMELS_US_NAME, SSM_SMAP_NAME]:
        data_gap = [0, 0]
        fill_nan = ["no", "no"]
        n_output = 2
    elif targets == [Q_CAMELS_US_NAME, ET_MODIS_NAME, SSM_SMAP_NAME]:
        data_gap = [0, 2, 0]
        fill_nan = ["no", "mean", "no"]
        n_output = 3
    else:
        raise NotImplementedError("We don't have such an exp")
    if gage_id_file is None:
        gage_id = "ALL"
    else:
        gage_id = pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values.tolist()
    if scaler_params is None:
        # same with config.py file
        scaler_params = {
            "basin_norm_cols": [
                Q_CAMELS_US_NAME,
                "streamflow",
                Q_CAMELS_CC_NAME,
                "qobs",
            ],
            "gamma_norm_cols": [
                PRCP_DAYMET_NAME,
                "pr",
                PRCP_NLDAS_NAME,
                "pre",
                # pet may be negative, but we set negative as 0 because of gamma_norm_cols
                # https://earthscience.stackexchange.com/questions/12031/does-negative-reference-evapotranspiration-make-sense-using-fao-penman-monteith
                "pet",
                PET_NLDAS_NAME,
                ET_MODIS_NAME,
                "LE",
                PET_MODIS_NAME,
                "PLE",
                "GPP",
                "Ec",
                "Es",
                "Ei",
                "ET_water",
                "ET_sum",
                SSM_SMAP_NAME,
                "susm",
                "smp",
                "ssma",
                "susma",
            ],
        }
    if loss_func == "MultiOutWaterBalanceLoss":
        loss_param = {
            "loss_funcs": "RMSESum",
            "data_gap": data_gap,
            "item_weight": loss_weights,
            "alpha": alpah,
        }
    else:
        loss_param = {
            "loss_funcs": "RMSESum",
            "data_gap": data_gap,
            "item_weight": loss_weights,
        }
    args = cmd(
        sub=project_name,
        source_path=[
            os.path.join(definitions.DATASET_DIR, "camelsflowet"),
            os.path.join(definitions.DATASET_DIR, "modiset4camels"),
            os.path.join(definitions.DATASET_DIR, "camels", "camels_us"),
            os.path.join(definitions.DATASET_DIR, "nldas4camels"),
            os.path.join(definitions.DATASET_DIR, "smap4camels"),
        ],
        source="CAMELS_FLOW_ET",
        download=0,
        ctx=[0],
        model_name="KuaiLSTMMultiOut",
        model_param={
            "n_input_features": 23,
            "n_output_features": n_output,
            "n_hidden_states": 256,
            "layer_hidden_size": 128,
        },
        loss_func=loss_func,
        loss_param=loss_param,
        cache_write=1,
        batch_size=100,
        rho=365,
        var_t=VAR_T_CHOSEN_FROM_NLDAS,
        var_t_type=[NLDAS_NAME],
        var_out=targets,
        train_period=train_period,
        test_period=test_period,
        data_loader="StreamflowDataModel",
        scaler="DapengScaler",
        scaler_params=scaler_params,
        train_epoch=300,
        te=300,
        save_epoch=20,
        fill_nan=fill_nan,
        n_output=n_output,
        gage_id=gage_id,
        stat_dict_file=stat_dict_file,
    )
    predict_in_test_period_with_model(
        args, weight_path=weight_path, cache_cfg_dir=cache_path
    )


def run_mtl_camels_flow_et(
    target_exp,
    var_c=VAR_C_CHOSEN_FROM_CAMELS_US,
    var_t=VAR_T_CHOSEN_FROM_NLDAS,
    train_period=None,
    test_period=None,
    weight_ratio=None,
    gage_id_file=os.path.join(
        definitions.RESULT_DIR,
        "camels_us_mtl_2001_2021_flow_screen.csv",
    ),
    cache_dir=None,
    random_seed=1234,
    ctx=None,
    loss_func="MultiOutLoss",
):
    if ctx is None:
        ctx = [0]
    if train_period is None:
        train_period = ["2001-10-01", "2011-10-01"]
    if test_period is None:
        test_period = ["2011-10-01", "2016-10-01"]
    if weight_ratio is None:
        weight_ratio = [0.5, 0.5]
    if loss_func == "MultiOutLoss":
        loss_param = {
            "loss_funcs": "RMSESum",
            "data_gap": [0, 2],
            "device": ctx,
            "item_weight": weight_ratio,
        }
    elif loss_func == "UncertaintyWeights":
        loss_param = {
            "loss_funcs": "RMSESum",
            "data_gap": [0, 2],
            "device": ctx,
        }
    else:
        raise NotImplementedError("No such loss function")
    weight_path_dir = os.path.join(definitions.RESULT_DIR, target_exp)
    try:
        weight_path = (
            get_lastest_weight_path(weight_path_dir)
            if os.path.exists(weight_path_dir)
            else None
        )
    except Exception as e:
        weight_path = None
    config_data = default_config_file()
    args = cmd(
        sub=f"camels/{target_exp}",
        source="CAMELS_FLOW_ET",
        source_path=[
            os.path.join(definitions.DATASET_DIR, "camelsflowet"),
            os.path.join(definitions.DATASET_DIR, "modiset4camels"),
            os.path.join(definitions.DATASET_DIR, "camels", "camels_us"),
            os.path.join(definitions.DATASET_DIR, "nldas4camels"),
            os.path.join(definitions.DATASET_DIR, "smap4camels"),
        ],
        download=0,
        ctx=ctx,
        model_name="KuaiLSTMMultiOut",
        model_param={
            "n_input_features": len(var_c) + len(var_t),
            "n_output_features": 2,
            "n_hidden_states": 256,
            "layer_hidden_size": 128,
        },
        weight_path=weight_path,
        loss_func=loss_func,
        loss_param=loss_param,
        cache_read=1,
        cache_write=1,
        batch_size=100,
        rho=365,
        var_t=var_t,
        var_c=var_c,
        var_t_type=[NLDAS_NAME],
        var_out=[Q_CAMELS_US_NAME, ET_MODIS_NAME],
        train_period=train_period,
        test_period=test_period,
        opt="Adadelta",
        rs=random_seed,
        data_loader="StreamflowDataModel",
        scaler="DapengScaler",
        n_output=2,
        train_epoch=300,
        save_epoch=20,
        te=300,
        # train_epoch=2,
        # save_epoch=1,
        # te=2,
        fill_nan=["no", "mean"],
        gage_id_file=gage_id_file,
    )
    update_cfg(config_data, args)
    if cache_dir is not None:
        # train_data_dict.json is a flag for cache existing
        if not os.path.exists(os.path.join(cache_dir, "train_data_dict.json")):
            cache_dir = None
        else:
            config_data["data_params"]["cache_path"] = cache_dir
            config_data["data_params"]["cache_read"] = True
            config_data["data_params"]["cache_write"] = False
    train_and_evaluate(config_data)
    print("All processes are finished!")


def plot_mtl_results_map(show_results, category_names, markers, save_file_path):
    # get index of the best result in all cases (single or multiple task) for each basin
    best_results = np.array(show_results).max(axis=0)
    s_m_indices = np.array(show_results).argmax(axis=0)
    # get those sites whose best NSE>0
    nse_pos_idx = np.argwhere(best_results > 0).flatten()
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels", "camels_us"))
    lat_lon = camels.read_constant_cols(
        camels.camels_sites["gauge_id"].values, ["gauge_lat", "gauge_lon"]
    )
    idx_lst_plot = [
        np.argwhere(s_m_indices == i).flatten()
        for i in range(min(s_m_indices), max(s_m_indices) + 1)
    ]
    idx_lst_plot_final = [np.intersect1d(nse_pos_idx, lst) for lst in idx_lst_plot]
    plot_map_carto(
        best_results,
        lat=lat_lon[:, 0],
        lon=lat_lon[:, 1],
        # pertile_range=[0, 100],
        value_range=[0, 1],
        idx_lst=idx_lst_plot_final,
        category_names=category_names,
        markers=markers,
        marker_size=10,
        legend_font_size=10,
    )
    FIGURE_DPI = 600
    plt.savefig(
        save_file_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )


def plot_multi_metrics_for_stl_mtl(exps_q_et, result_dir, var_obj="flow"):
    show_inds = ["Bias", "RMSE", "Corr", "NSE"]
    plt.rc("axes", labelsize=16)
    plt.rc("ytick", labelsize=12)
    FIGURE_DPI = 600
    assert len(exps_q_et) == 2
    exps_metrices_results = []
    for ind in show_inds:
        if var_obj == "flow":
            exp_metric_results = read_multi_single_exps_results(
                exps_q_et, metric=ind, ensemble=-1
            )
        else:
            exp_metric_results = read_multi_single_exps_results(
                exps_q_et, metric=ind, var_idx=1, single_is_flow=False, ensemble=-1
            )
        exps_metrices_results.append([exp_metric_results[0], exp_metric_results[1]])
    plot_boxes_matplotlib(
        exps_metrices_results,
        label1=show_inds,
        label2=["单变量", "多变量"],
        colorlst=["#d62728", "#1f77b4"],
        figsize=(10, 5),
        subplots_adjust_wspace=0.35,
        median_font_size="xx-small",
    )
    plt.savefig(
        os.path.join(
            result_dir,
            "mtl_" + var_obj + "_test_all_metrices_boxes.png",
        ),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )


def run_mtl_camels_cc_flow_et(
    target_exp,
    var_c=VAR_C_CHOSEN_FROM_CAMELS_CC,
    var_t=VAR_T_CHOSEN_FROM_ERA5LAND,
    train_period=None,
    test_period=None,
    weight_ratio=None,
    gage_id_file=os.path.join(
        definitions.ROOT_DIR,
        "hydroSPB",
        "example",
        "camels",
        "camels_cc_2014_2020_flow_screen.csv",
    ),
):
    if train_period is None:
        train_period = ["2014-10-01", "2019-10-01"]
    if test_period is None:
        test_period = ["2019-10-01", "2020-10-01"]
    if weight_ratio is None:
        weight_ratio = [0.5, 0.5]
    weight_path_dir = os.path.join(
        definitions.ROOT_DIR, "hydroSPB", "example", "camels", target_exp
    )
    weight_path = get_lastest_weight_path(weight_path_dir)
    config_data = default_config_file()
    args = cmd(
        sub=f"camels/{target_exp}",
        source="CAMELS",
        source_region="CC",
        source_path=os.path.join(definitions.DATASET_DIR, "camels", "camels_cc"),
        download=0,
        ctx=[0],
        model_name="KuaiLSTMMultiOut",
        model_param={
            "model_name": "multi_output_tl",
            "n_input_features": len(var_c) + len(var_t),
            "n_output_features": 2,
            "n_hidden_states": 256,
            "layer_hidden_size": 128,
        },
        weight_path=weight_path,
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 2],
            "device": [0],
            "item_weight": weight_ratio,
        },
        cache_read=1,
        cache_write=1,
        batch_size=10,
        rho=365,
        var_t=var_t,
        var_c=var_c,
        var_t_type=[ERA5LAND_NAME],
        var_out=[Q_CAMELS_CC_NAME, ET_MODIS_NAME],
        train_period=train_period,
        test_period=test_period,
        opt="Adadelta",
        rs=1234,
        data_loader="StreamflowDataModel",
        scaler="DapengScaler",
        n_output=2,
        train_epoch=100,
        save_epoch=10,
        te=100,
        fill_nan=["no", "mean"],
        gage_id_file=gage_id_file,
    )
    update_cfg(config_data, args)
    train_and_evaluate(config_data)
    print("All processes are finished!")
