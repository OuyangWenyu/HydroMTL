"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2022-12-24 20:07:40
LastEditors: Wenyu Ouyang
Description: Testing functions for hydroDL models
FilePath: /HydroSPB/hydroSPB/hydroDL/evaluator.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import time
import os
import logging
from typing import Dict, Tuple
from functools import reduce
import numpy as np
import pandas as pd
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from hydroSPB.data.loader.data_loaders import (
    DplDataModel,
)
from hydroSPB.data.loader.dataloaders4test import (
    TestDataModel,
    TestDplDataModel,
    TestDplDataset,
)
from hydroSPB.data.loader.data_sets import DplDataset
from hydroSPB.data.loader.xr_dataloader import XarrayDataModel, TestXarrayDataModel
from hydroSPB.hydroDL.hydros.pbm_config import MODEL_PARAM_TEST_WAY
from hydroSPB.utils.hydro_stat import stat_error
from hydroSPB.utils import hydro_utils

from hydroSPB.hydroDL.model_dict_function import sequence_first_model_lst
from hydroSPB.hydroDL.time_model import PyTorchForecast
from hydroSPB.visual.explain_model_output import (
    deep_explain_model_summary_plot,
    deep_explain_model_heatmap,
)
from hydroSPB.hydroDL.training_utils import get_the_device


def evaluate_model(model: PyTorchForecast) -> Tuple[Dict, np.array, np.array]:
    """
    A function to evaluate a model, called at end of training.

    Parameters
    ----------
    model
        the DL model class

    Returns
    -------
    tuple[dict, np.array, np.array]
        eval_log, denormalized predictions and observations
    """
    data_params = model.params["data_params"]
    # types of observations
    target_col = model.params["data_params"]["target_cols"]
    evaluation_metrics = model.params["evaluate_params"]["metrics"]
    # fill_nan: "no" means ignoring the NaN value;
    #           "sum" means calculate the sum of the following values in the NaN locations.
    #           For example, observations are [1, nan, nan, 2], and predictions are [0.3, 0.3, 0.3, 1.5].
    #           Then, "no" means [1, 2] v.s. [0.3, 1.5] while "sum" means [1, 2] v.s. [0.3 + 0.3 + 0.3, 1.5].
    #           If it is a str, then all target vars use same fill_nan method;
    #           elif it is a list, each for a var
    fill_nan = model.params["evaluate_params"]["fill_nan"]
    # save result here
    eval_log = {}

    # test the trained model
    test_epoch = model.params["evaluate_params"]["test_epoch"]
    train_epoch = model.params["training_params"]["epochs"]
    if test_epoch != train_epoch:
        # Generally we use same epoch for train and test, but sometimes not
        # TODO: better refactor this part, because sometimes we save multi models for multi hyperparameters
        model_filepath = model.params["data_params"]["test_path"]
        model.model = model.load_model(
            model.params["model_params"]["model_name"],
            model.params["model_params"],
            weight_path=os.path.join(model_filepath, f"model_Ep{str(test_epoch)}.pth"),
        )
    pred, obs, test_data = infer_on_torch_model(model)
    print("Un-transforming data")
    preds_np = test_data.inverse_scale(pred)
    obss_np = test_data.inverse_scale(obs)

    #  Then evaluate the model metrics
    if type(fill_nan) is list:
        if len(fill_nan) != len(target_col):
            raise Exception("length of fill_nan must be equal to target_col's")
    for i in range(len(target_col)):
        if type(fill_nan) is str:
            inds = stat_error(obss_np[:, :, i], preds_np[:, :, i], fill_nan)
        else:
            inds = stat_error(obss_np[:, :, i], preds_np[:, :, i], fill_nan[i])
        for evaluation_metric in evaluation_metrics:
            eval_log[evaluation_metric + " of " + target_col[i]] = inds[
                evaluation_metric
            ]

    # Finally, try to explain model behaviour using shap
    # TODO: SHAP has not been tested
    is_shap = False
    if is_shap:
        deep_explain_model_summary_plot(
            model, test_data, data_params["t_range_test"][0]
        )
        deep_explain_model_heatmap(model, test_data, data_params["t_range_test"][0])

    return eval_log, preds_np, obss_np


def infer_on_torch_model(
    model: PyTorchForecast,
) -> Tuple[torch.Tensor, torch.Tensor, TestDataModel]:
    """
    Function to handle both test evaluation and inference on a test data-frame.
    """
    data_params = model.params["data_params"]
    device = get_the_device(model.params["training_params"]["device"])
    model.model.eval()
    if type(model.test_data) is XarrayDataModel:
        test_dataset = TestXarrayDataModel(model.test_data)
        all_data = test_dataset.load_test_data()
        pred = xrds_predictions(
            model, test_dataset, *all_data[:-1], device=device, data_params=data_params
        )
    elif type(model.test_data) in [DplDataModel, DplDataset]:
        if type(model.test_data) == DplDataModel:
            test_dataset = TestDplDataModel(model.training, model.test_data)
        else:
            test_dataset = TestDplDataset(model.training, model.test_data)
        all_data = test_dataset.load_test_data()
        pred = dpl_model_predictions(
            model, test_dataset, *all_data, device=device, data_params=data_params
        )
    else:
        test_dataset = TestDataModel(model.test_data)
        all_data = test_dataset.load_test_data()
        pred = generate_predictions(
            model, test_dataset, *all_data[:-1], device=device, data_params=data_params
        )
    if pred.shape[1] != all_data[-1].shape[1]:
        # it means we use an Nto1 mode model, so cutoff some previous data for observations to be comparable
        return pred, all_data[-1][:, test_dataset.test_data.rho - 1 :, :], test_dataset
    return pred, all_data[-1], test_dataset


def dpl_model_predictions(
    ts_model: PyTorchForecast,
    test_dataset: TestDplDataModel,  # just keep consistent with other functions
    *args,
    device: torch.device,
    data_params: dict,
) -> np.ndarray:
    model = ts_model.model
    model.train(mode=False)
    if "tl_tag" in model.__dict__.keys():
        dl_model = model.tl_part.dl_model
        pb_model = model.tl_part.pb_model
        param_func = model.tl_part.param_func
        param_test_way = model.tl_part.param_test_way
        if param_test_way == MODEL_PARAM_TEST_WAY["time_varying"]:
            param_var_index = model.tl_part.param_var_index
    else:
        dl_model = model.dl_model
        pb_model = model.pb_model
        param_func = model.param_func
        param_test_way = model.param_test_way
        if param_test_way == MODEL_PARAM_TEST_WAY["time_varying"]:
            param_var_index = model.param_var_index

    # default setting is for final parameter of training period
    x_norm = args[0]
    c_norm = args[1]
    y_norm = args[2]
    x = args[3]
    c = args[4]

    if param_test_way == MODEL_PARAM_TEST_WAY["time_scroll"]:
        # scroll prediction
        x_norm = args[5]
        c_norm = args[6]
        y_norm = args[7]
    elif param_test_way == MODEL_PARAM_TEST_WAY["time_varying"]:
        # var means testing-period variable parameters will be used in the testing period,
        # NOTE: constant parameters are also generated in the testing period as our models are all open-loop
        x_norm = args[5]
        c_norm = args[6]
    elif param_test_way == MODEL_PARAM_TEST_WAY["final_period"]:
        # testing-period final parameters will be used in the testing period
        # because we are open-loop model, so it is reasonable to use the final parameters of testing period
        x_norm = args[5]
        c_norm = args[6]

    # generate parameters: ngrid is same in x_norm and x, but nt could be different
    ngrid, nt, _ = x_norm.shape
    nc = c_norm.shape[-1]

    i_s = np.arange(0, ngrid, data_params["batch_size"])
    i_e = np.append(i_s[1:], ngrid)

    params_lst = []
    for i in range(0, len(i_s)):
        x_norm_temp = x_norm[i_s[i] : i_e[i], :, :]
        y_norm_temp = y_norm[i_s[i] : i_e[i], :, :]
        if nc == 0:
            c_norm_temp = None
        else:
            c_norm_temp = np.repeat(
                np.reshape(c_norm[i_s[i] : i_e[i], :], [i_e[i] - i_s[i], 1, nc]),
                nt,
                axis=1,
            )
        if data_params["target_as_input"]:
            if nc == 0:
                norm_temp = torch.from_numpy(
                    np.swapaxes(np.concatenate([x_norm_temp, y_norm_temp], 2), 1, 0)
                ).float()
            else:
                norm_temp = torch.from_numpy(
                    np.swapaxes(
                        np.concatenate([x_norm_temp, c_norm_temp, y_norm_temp], 2), 1, 0
                    )
                ).float()
        else:
            if data_params["constant_only"]:
                norm_temp = torch.from_numpy(c_norm[i_s[i] : i_e[i], :]).float()
            else:
                if nc == 0:
                    norm_temp = torch.from_numpy(np.swapaxes(x_norm_temp, 1, 0)).float()
                else:
                    norm_temp = torch.from_numpy(
                        np.swapaxes(np.concatenate([x_norm_temp, c_norm_temp], 2), 1, 0)
                    ).float()
        norm_temp = norm_temp.to(device)
        if "tl_tag" in model.__dict__.keys():
            # the layer before tl_part must be former_linear
            gen = dl_model(model.former_linear(norm_temp))
        else:
            gen = dl_model(norm_temp)
        # just get one-period values, here we use the final period's values
        params_lst.append(gen)
    # All models are sequence-first, so two cases: [seq, batch, feature] or [batch, feature]. Here concat in batch's dim
    pb_params = reduce(lambda a, b: torch.cat((a, b), dim=-2), params_lst)
    # Firstly, we limit the range of intermediate parameters; then what we will deal with is the real parameters
    # we set all params' values in [0, 1] and will scale them when forwarding
    if param_func == "sigmoid":
        pb_params_ = F.sigmoid(pb_params)
    elif param_func == "clamp":
        pb_params_ = torch.clamp(pb_params, min=0.0, max=1.0)
    else:
        raise NotImplementedError(
            "We don't provide this way to limit parameters' range!! Please choose sigmoid or clamp"
        )

    # Predict
    y_out_list = []
    param_list = []
    for i in range(0, len(i_s)):
        print("batch {}".format(i))

        x_temp = torch.from_numpy(np.swapaxes(x[i_s[i] : i_e[i], :], 1, 0)).float()
        x_temp = x_temp.to(device)

        if param_test_way in [
            MODEL_PARAM_TEST_WAY["final_period"],
            MODEL_PARAM_TEST_WAY["final_train_period"],
        ]:
            if data_params["constant_only"]:
                params = pb_params_[i_s[i] : i_e[i], :]
            else:
                # just like training
                params = pb_params_[-1, i_s[i] : i_e[i], :]
        elif param_test_way == MODEL_PARAM_TEST_WAY["mean_all_period"]:
            if data_params["constant_only"]:
                logging.warning(
                    "If only attributes data are used, 'mean_time' mode is not supported as no time-sequence data."
                    "So we directly use params like 'final' mode"
                )
                model.param_test_way = "final"
                params = pb_params_[i_s[i] : i_e[i], :]
            else:
                # we should use parameters for periods after rho-1 because of the training process
                params = torch.mean(
                    pb_params_[
                        data_params["forecast_history"] - 1 :, i_s[i] : i_e[i], :
                    ],
                    dim=0,
                )
        elif param_test_way == MODEL_PARAM_TEST_WAY["mean_all_basin"]:
            if data_params["constant_only"]:
                params = torch.tile(
                    torch.mean(pb_params_[i_s[i] : i_e[i], :], dim=0),
                    (i_e[i] - i_s[i], 1),
                )
            else:
                params = torch.tile(
                    torch.mean(pb_params_[-1:, :, :], dim=1), (i_e[i] - i_s[i], 1)
                )
        elif param_test_way in [
            MODEL_PARAM_TEST_WAY["time_varying"],
            MODEL_PARAM_TEST_WAY["time_scroll"],
        ]:
            # for open-loop model, we can use parameters generated in test period
            if data_params["constant_only"]:
                raise ArithmeticError("Cannot constant_only when use dynamic mode")
            else:
                params = pb_params_[:, i_s[i] : i_e[i], :]
        else:
            raise NotImplementedError(
                "We don't provide this testing way; please choose one from: 'final', 'mean_time', 'mean_basin'!!"
            )
        if param_test_way == MODEL_PARAM_TEST_WAY["time_scroll"]:
            # TODO: not fully tested yet
            y_p_tmp_lst = []
            for i in range(pb_model.warmup_length, params.shape[0]):
                # use kernel_size because when calculateing conv we need a data whose size is not smaller than uh_size
                y_p_tmp = pb_model(
                    x_temp[:, :, : pb_model.feature_size], params[i, :, :]
                )
                chosen_y_p = y_p_tmp[
                    i - pb_model.warmup_length : i - pb_model.warmup_length + 1,
                    :,
                    :,
                ]
                y_p_tmp_lst.append(chosen_y_p.detach().cpu().numpy())
                torch.cuda.empty_cache()
            y_p = reduce(lambda a, b: np.concatenate((a, b), axis=0), y_p_tmp_lst)
            y_out_list.append(y_p.swapaxes(0, 1))
            params_out = (
                params[pb_model.warmup_length :, :, :]
                .detach()
                .cpu()
                .numpy()
                .swapaxes(0, 1)
            )
            param_list.append(params_out)
        else:
            y_p = pb_model(x_temp[:, :, : pb_model.feature_size], params)
            if type(y_p) is tuple:
                y_p = reduce(lambda a, b: torch.cat((a, b), dim=-1), y_p)
            y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)
            y_out_list.append(y_out)
            if param_test_way == MODEL_PARAM_TEST_WAY["time_varying"]:
                param_list.append(
                    params[pb_model.warmup_length :, :, :]
                    .detach()
                    .cpu()
                    .numpy()
                    .swapaxes(0, 1)
                )
            else:
                param_list.append(params.detach().cpu().numpy())

    model.zero_grad()
    torch.cuda.empty_cache()
    pred = reduce(lambda a, b: np.vstack((a, b)), y_out_list)
    param_output = reduce(lambda a, b: np.vstack((a, b)), param_list)
    sites = ts_model.params["data_params"]["object_ids"]
    params_names = pb_model.params_names
    if param_test_way in [
        MODEL_PARAM_TEST_WAY["time_varying"],
        MODEL_PARAM_TEST_WAY["time_scroll"],
    ]:
        pb_param_dir = os.path.join(
            ts_model.params["data_params"]["test_path"],
            "pb_params_" + str(int(time.time())),
        )
        if not os.path.isdir(pb_param_dir):
            os.makedirs(pb_param_dir)
        time_series = hydro_utils.t_range_days(
            ts_model.params["data_params"]["t_range_test"]
        )
        for j in range(len(params_names)):
            save_param_file = os.path.join(pb_param_dir, params_names[j] + ".csv")
            params_df = pd.DataFrame(
                param_output[:, :, j],
                columns=time_series[pb_model.warmup_length :],
                index=sites,
            )
            params_df.to_csv(save_param_file, index_label="GAGE_ID")
        return pred
    param_save_dir = ts_model.params["data_params"]["test_path"]
    params_df = pd.DataFrame(param_output, columns=params_names, index=sites)
    save_param_file = os.path.join(
        param_save_dir, "pb_params_" + str(int(time.time())) + ".csv"
    )
    params_df.to_csv(save_param_file, index_label="GAGE_ID")
    # save all params to see the pattern of change of params
    np.save(
        os.path.join(param_save_dir, "pb_params_np_" + str(int(time.time())) + ".npy"),
        pb_params_.detach().cpu().numpy(),
    )
    return pred


def xrds_predictions(
    ts_model: PyTorchForecast,
    test_data_model: TestDataModel,
    *args,
    device: torch.device,
    data_params: dict,
) -> np.ndarray:
    model = ts_model.model
    model.train(mode=False)
    if type(model) is DataParallel:
        if type(model.module) in sequence_first_model_lst:
            batch_first = False
        else:
            batch_first = True
    else:
        if type(model) in sequence_first_model_lst:
            batch_first = False
        else:
            batch_first = True
    x = args[0]
    ngrid = len(x)
    # cannot use x[0]["time"].size directly, because daymet data only have 365 days in a leap year
    nt = len(hydro_utils.t_range_days(test_data_model.test_data.t_range))

    y_out_list = []
    # loop for each basin and each time-slice (here cut test period to 2 parts)
    for i in range(0, ngrid):
        print("batch {}".format(i))
        one_batch = test_data_model.test_data.get_item(
            np.array([i, i]),
            np.array([0, int(nt / 2)]),
            int(nt / 2) + 1,
            batch_first=batch_first,
        )
        # Convert to CPU/GPU/TPU
        xy = [data_tmp.to(device) for data_tmp in one_batch]
        y_p = model(*xy[0:-1])
        if not batch_first:  # seq first
            y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)
        else:
            y_out = y_p.detach().cpu().numpy()
        # we split the time, so now concatenate them. notice when nt is odd, we ignore the first value of final piece
        if nt % 2 == 0:
            y_out = np.vstack(y_out).reshape(1, -1, 1)
        else:
            y_out = np.concatenate(
                [y_out[:-1]] + [y_out[-1][1:].reshape(1, -1, 1)], axis=1
            )
        y_out_list.append(y_out)

    model.zero_grad()
    torch.cuda.empty_cache()
    data_stack = reduce(
        lambda a, b: np.vstack((a, b)),
        list(map(lambda x: x.reshape(x.shape[0], x.shape[1]), y_out_list)),
    )
    pred = np.expand_dims(data_stack, axis=2)
    return pred


def generate_predictions(
    ts_model: PyTorchForecast,
    test_model: TestDataModel,
    *args,
    device: torch.device,
    data_params: dict,
    return_cell_state: bool = False,
) -> np.ndarray:
    """Perform Evaluation on the test (or valid) data.

    Parameters
    ----------
    ts_model : PyTorchForecast
        _description_
    test_model : TestDataModel
        _description_
    device : torch.device
        _description_
    data_params : dict
        _description_
    return_cell_state : bool, optional
        if True, time-loop evaluation for cell states, by default False
        NOTE: ONLY for LSTM models

    Returns
    -------
    np.ndarray
        _description_
    """
    model = ts_model.model
    model.train(mode=False)
    if type(model) in sequence_first_model_lst:
        seq_first = True
    else:
        seq_first = False
    if issubclass(type(test_model.test_data), Dataset):
        # TODO: not support return_cell_states yet
        # here the batch is just an index of lookup table, so any batch size could be chosen
        test_loader = DataLoader(
            test_model.test_data, batch_size=data_params["batch_size"], shuffle=False
        )
        test_preds = []
        with torch.no_grad():
            for i_batch, (xs, ys) in enumerate(test_loader):
                # here the a batch doesn't mean a basin; it is only an index in lookup table
                # for NtoN mode, only basin is index in lookup table, so the batch is same as basin
                # for Nto1 mode, batch is only an index
                if seq_first:
                    xs = xs.transpose(0, 1)
                    ys = ys.transpose(0, 1)
                xs = xs.to(device)
                ys = ys.to(device)
                output = model(xs)
                if type(output) is tuple:
                    others = output[1:]
                    # Convention: y_p must be the first output of model
                    output = output[0]
                if seq_first:
                    output = output.transpose(0, 1)
                test_preds.append(output.cpu().numpy())
            pred = reduce(lambda x, y: np.vstack((x, y)), test_preds)
        if pred.ndim == 2:
            # the ndim is 2 meaning we use an Nto1 mode
            # as lookup table is (basin 1's all time length, basin 2's all time length, ...)
            # params of reshape should be (basin size, time length)
            pred = pred.flatten().reshape(test_model.test_data.y.shape[0], -1, 1)

    else:
        x = args[0]
        c = args[1]
        z = None
        if len(args) == 3:
            z = args[2]
        ngrid, nt, nx = x.shape
        if c is not None:
            nc = c.shape[-1]

        i_s = np.arange(0, ngrid, data_params["batch_size"])
        i_e = np.append(i_s[1:], ngrid)

        y_out_list = []
        if return_cell_state:
            # all basins' cell states
            cs_out_lst = []
        for i in range(0, len(i_s)):
            # print("batch {}".format(i))
            x_temp = x[i_s[i] : i_e[i], :, :]

            if c is not None and c.shape[-1] > 0:
                c_temp = np.repeat(
                    np.reshape(c[i_s[i] : i_e[i], :], [i_e[i] - i_s[i], 1, nc]),
                    nt,
                    axis=1,
                )
                if seq_first:
                    xhTest = torch.from_numpy(
                        np.swapaxes(np.concatenate([x_temp, c_temp], 2), 1, 0)
                    ).float()
                else:
                    xhTest = torch.from_numpy(
                        np.concatenate([x_temp, c_temp], 2)
                    ).float()
            else:
                if seq_first:
                    xhTest = torch.from_numpy(np.swapaxes(x_temp, 1, 0)).float()
                else:
                    xhTest = torch.from_numpy(x_temp).float()
            xhTest = xhTest.to(device)
            with torch.no_grad():
                if z is not None:
                    # now only support z is 2d var
                    assert z.ndim == 2
                    if seq_first:
                        zTemp = z[i_s[i] : i_e[i], :]
                        zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
                    else:
                        zTest = torch.from_numpy(z[i_s[i] : i_e[i], :]).float()
                    zTest = zTest.to(device)
                    y_p = model(xhTest, zTest)
                else:
                    if return_cell_state:
                        cs_lst = []
                        for j in range(nt):
                            y_p_, (hs, cs) = model(
                                xhTest[0 : j + 1, :, :], return_h_c=True
                            )
                            cs_lst.append(cs)
                        cs_cat_lst = torch.cat(cs_lst, dim=0)
                    y_p = model(xhTest)
                if type(y_p) is tuple:
                    others = y_p[1:]
                    # Convention: y_p must be the first output of model
                    y_p = y_p[0]
                if seq_first:
                    y_out = y_p.detach().cpu().numpy().swapaxes(0, 1)
                else:
                    y_out = y_p.detach().cpu().numpy()

                y_out_list.append(y_out)
                if return_cell_state:
                    if seq_first:
                        cs_out = cs_cat_lst.detach().cpu().numpy().swapaxes(0, 1)
                    else:
                        cs_out = cs_cat_lst.detach().cpu().numpy()
                    cs_out_lst.append(cs_out)
        # model.zero_grad()
        torch.cuda.empty_cache()
        pred = reduce(lambda a, b: np.vstack((a, b)), y_out_list)
        if return_cell_state:
            cell_state = reduce(lambda a, b: np.vstack((a, b)), cs_out_lst)
            np.save(
                os.path.join(data_params["test_path"], "cell_states.npy"), cell_state
            )
            return pred, cell_state

    return pred
