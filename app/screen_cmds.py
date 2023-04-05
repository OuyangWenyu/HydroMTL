"""
Author: Wenyu Ouyang
Date: 2022-04-27 10:54:32
LastEditTime: 2022-08-18 15:59:08
LastEditors: Wenyu Ouyang
Description: Generate commands to run scripts in Linux Screen
FilePath: /HydroSPB/hydroSPB/app/screen_cmds.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import argparse
import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions


def scripts4mtlcv(
    screens, tasks, train_periods, test_periods, data_gaps, ctxs, weights, kfolds, paths
):
    """Generate scripts for Cross Validation exps in MTL

    Note: Now only support linux

    Parameters
    ----------
    screens : list
        such as ["exp431", "exp432", ...]
    tasks: list
        ["ET", "ET"]
    train_periods: list
        [["2001-10-01", "2011-10-01"], ...]
    test_periods: list
        [["2011-10-01", "2021-10-01"], ...]
    data_gaps: list
        [2, 2]
    ctxs: list
        gpus, [0,0,...]
    weights: list
        weights used for losses, [[0.5, 0.5],...]
    kfolds: list
        the i-th index of k-fold, such as [0,1,2] in 3-fold
    paths: list
        cache list

    Returns
    -------
    list
        scripts
    """
    scripts = []
    sep = os.sep
    code_path = sep.join(
        os.path.realpath(__file__).split(sep)[:-1] + ["train_evaluate.py"]
    )
    split_num = str(len(kfolds))
    dir_root = definitions.ROOT_DIR
    dir_data = definitions.DATASET_DIR
    for i in range(len(screens)):
        weight = weights[i]
        path = paths[i]
        task = tasks[i]
        data_gap = data_gaps[i]
        train_period = train_periods[i]
        test_period = test_periods[i]
        if data_gap == 0:
            fill_nan = "no"
        elif data_gap == 1:
            fill_nan = "sum"
        elif data_gap == 2:
            fill_nan = "mean"
        else:
            raise NotImplementedError("no such data_gap setting ye")
        if path is not None:
            script = (
                'python %(code_path)s --sub camels/%(exp)s --source CAMELS_FLOW_ET --source_path %(dir_data)s/camelsflowet %(dir_data)s/modiset4camels %(dir_data)s/camels/camels_us %(dir_data)s/nldas4camels %(dir_data)s/smap4camels --download 0 --ctx %(ctx)d --model_name KuaiLSTMMultiOut --model_param {%(slash)s"n_input_features%(slash)s":23%(slash)s,%(slash)s"n_output_features%(slash)s":2%(slash)s,%(slash)s"n_hidden_states%(slash)s":256%(slash)s,%(slash)s"layer_hidden_size%(slash)s":128} --loss_func MultiOutLoss --loss_param {%(slash)s"loss_funcs%(slash)s":%(slash)s"RMSESum%(slash)s"%(slash)s,%(slash)s"data_gap%(slash)s":[0%(slash)s,%(data_gap)d]%(slash)s,%(slash)s"device%(slash)s":[%(ctx)d]%(slash)s,%(slash)s"item_weight%(slash)s":[%(w1)f%(slash)s,%(w2)f]} --cache_read 1 --cache_path %(path)s --batch_size 100 --rho 365 --var_t total_precipitation potential_evaporation temperature specific_humidity shortwave_radiation potential_energy --var_t_type nldas --var_out usgsFlow %(task)s --train_period %(train1)s %(train2)s --test_period %(test1)s %(test2)s --opt Adadelta --rs 1234 --data_loader StreamflowDataModel --scaler DapengScaler --n_output 2 --train_epoch 300 --save_epoch 10 --te 300 --fill_nan no %(fill_nan)s --gage_id_file %(dir_root)s/hydroSPB/example/camels/exp43_kfold%(split_num)s/camels_train_kfold%(idx)s.csv'
                % {
                    "dir_root":dir_root,
                    "dir_data":dir_data,
                    "code_path": code_path,
                    "exp": screens[i],
                    "ctx": ctxs[i],
                    # need a "\" in linux
                    "slash": "\\",
                    "w1": weight[0],
                    "w2": weight[1],
                    "idx": str(kfolds[i]),
                    "path": path,
                    "task": task,
                    "train1": train_period[0],
                    "train2": train_period[1],
                    "test1": test_period[0],
                    "test2": test_period[1],
                    "split_num": split_num,
                    "data_gap": data_gap,
                    "fill_nan":fill_nan
                }
            )
        else:
            script = (
                'python %(code_path)s --sub camels/%(exp)s --source CAMELS_FLOW_ET --source_path %(dir_data)s/camelsflowet %(dir_data)s/modiset4camels %(dir_data)s/camels/camels_us %(dir_data)s/nldas4camels %(dir_data)s/smap4camels --download 0 --ctx %(ctx)d --model_name KuaiLSTMMultiOut --model_param {%(slash)s"n_input_features%(slash)s":23%(slash)s,%(slash)s"n_output_features%(slash)s":2%(slash)s,%(slash)s"n_hidden_states%(slash)s":256%(slash)s,%(slash)s"layer_hidden_size%(slash)s":128} --loss_func MultiOutLoss --loss_param {%(slash)s"loss_funcs%(slash)s":%(slash)s"RMSESum%(slash)s"%(slash)s,%(slash)s"data_gap%(slash)s":[0%(slash)s,%(data_gap)d]%(slash)s,%(slash)s"device%(slash)s":[%(ctx)d]%(slash)s,%(slash)s"item_weight%(slash)s":[%(w1)f%(slash)s,%(w2)f]} --cache_read 1 --cache_write 1 --batch_size 100 --rho 365 --var_t total_precipitation potential_evaporation temperature specific_humidity shortwave_radiation potential_energy --var_t_type nldas --var_out usgsFlow %(task)s --train_period %(train1)s %(train2)s --test_period %(test1)s %(test2)s --opt Adadelta --rs 1234 --data_loader StreamflowDataModel --scaler DapengScaler --n_output 2 --train_epoch 300 --save_epoch 10 --te 300 --fill_nan no %(fill_nan)s --gage_id_file %(dir_root)s/hydroSPB/example/camels/exp43_kfold%(split_num)s/camels_train_kfold%(idx)s.csv'
                % {
                    "dir_root":dir_root,
                    "dir_data":dir_data,
                    "code_path": code_path,
                    "exp": screens[i],
                    "ctx": ctxs[i],
                    # need a "\" in linux
                    "slash": "\\",
                    "w1": weight[0],
                    "w2": weight[1],
                    "idx": str(kfolds[i]),
                    "task": task,
                    "train1": train_period[0],
                    "train2": train_period[1],
                    "test1": test_period[0],
                    "test2": test_period[1],
                    "split_num": split_num,
                    "data_gap": data_gap,
                    "fill_nan":fill_nan
                }
            )
        scripts.append(script)
    return scripts


def run_in_screen(args):
    screens = args.screen
    tasks = args.task
    train_periods = args.train
    test_periods = args.test
    data_gaps = args.gap
    ctxs = args.ctx
    weights = args.weight
    kfolds = args.kfold
    path = args.path
    if path is None:
        paths = [None] * len(kfolds)
    else:
        paths = [
            os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", "camels", tmp)
            if tmp is not None
            else None
            for tmp in path
        ]
    scripts = scripts4mtlcv(
        screens=screens,
        tasks=tasks,
        train_periods=train_periods,
        test_periods=test_periods,
        data_gaps=data_gaps,
        ctxs=ctxs,
        weights=weights,
        kfolds=kfolds,
        paths=paths,
    )
    for k in range(len(scripts)):
        # -dmS means "Start as daemon: Screen session in detached mode"
        # https://stackoverflow.com/questions/27091995/using-linux-screen-command-from-python
        # becasue we cannot get one backslash in a string, the command cannot be executed successfully:
        # subprocess.call(["screen", "-dmS", screens[k], scripts[k]])
        # so we just print the command, you can copy and paste them in your bash
        print(
            "screen"
            + " -S "
            + screens[k]
            + "\n"
            + "conda activate HydroSPB10"
            + "\n"
            + scripts[k]
        )


if __name__ == "__main__":
    print("Here are your commands:")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-S",
        dest="screen",
        help="screen IDs",
        nargs="+",
        type=str,
        default=["exp43007", "exp43008"],
    )
    parser.add_argument(
        "-T", dest="task", help="ET or ssm", nargs="+", type=str, default=["ET", "ET"],
    )
    parser.add_argument(
        "-G",
        dest="gap",
        help="data_gap, ET is 2 / ssm is 0",
        nargs="+",
        type=int,
        default=[2, 2],
    )
    parser.add_argument(
        "-R",
        dest="train",
        help="training periods, such as ['2001-10-01', '2011-10-01']",
        nargs="+",
        type=str,
        action="append",
    )
    parser.add_argument(
        "-E",
        dest="test",
        help="testing periods, such as ['2011-10-01', '2021-10-01']",
        nargs="+",
        type=str,
        action="append",
    )
    parser.add_argument(
        "-C", dest="ctx", help="CUDA IDs", nargs="+", type=int, default=[0, 1]
    )
    # https://stackoverflow.com/questions/53712889/python-argparse-with-list-of-lists
    parser.add_argument(
        "-W",
        dest="weight",
        help="weights of losses as a list",
        nargs="+",
        type=float,
        action="append",
    )
    parser.add_argument(
        "-K",
        dest="kfold",
        help="indices of k-fold exps as a list",
        nargs="+",
        type=int,
        default=[0, 1],
    )
    parser.add_argument(
        "-P", dest="path", help="path of data cache", nargs="+", type=str, default=None
    )
    args = parser.parse_args()
    run_in_screen(args)
