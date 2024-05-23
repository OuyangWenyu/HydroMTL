"""
Author: Wenyu Ouyang
Date: 2022-12-12 17:01:00
LastEditTime: 2024-05-23 13:22:27
LastEditors: Wenyu Ouyang
Description: Analysis for one-basin experiments DL models
FilePath: \HydroMTL\scripts\weights_visual_analysis.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import matplotlib
import numpy as np
from scipy.spatial.distance import cosine
import itertools
from tqdm import tqdm
from matplotlib import cm, colors
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydromtl.data.source.data_camels import Camels
from scripts.streamflow_utils import read_tb_log, plot_ts_for_basin
from hydromtl.visual.plot_stat import plot_ts

# all figs from tensorboard were ploted according to: https://tbparse.readthedocs.io/en/latest/notebooks/gallery-pytorch.html
# Supress tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
ID_FILE_PATH = os.path.join(
    definitions.RESULT_DIR, "camels_us_mtl_2001_2021_flow_screen.csv"
)
basin_ids = pd.read_csv(ID_FILE_PATH, dtype={"GAGE_ID": str})["GAGE_ID"].tolist()
two_vars = ["Q", "ET"]
camels = Camels(
    os.path.join(definitions.DATASET_DIR, "camels", "camels_us"),
)
basin_areas = camels.read_basin_area(basin_ids)
model_names = ["STL", "MTL"]
stl_mtl_q_exps = ["expstlq001", "expmtl003"]
stl_mtl_et_exps = ["expstlet001", "expmtl003"]
exps = [stl_mtl_q_exps, stl_mtl_et_exps]
# first dim is for case, second dim is for model
best_batchsize = [[100, 100], [100, 100]]
best_epoch = [[300, 300], [300, 300]]
all_epoch = [300, 300]


def read_layer_name_from_tb_hist(hist_cols):
    layer_names = []
    for col in hist_cols:
        if "counts" in col:
            layer_name = col.split("/")[0]
            if layer_name not in layer_names:
                layer_names.append(layer_name)
    return layer_names


def epochs_hist_for_chosen_layer(epoch_interval, layer_name, df_hist):
    df = pd.DataFrame()
    all_epochs = df_hist.shape[0]
    limit_uppers = []
    limit_lowers = []
    for i in range(0, all_epochs, epoch_interval):
        limits = df_hist[layer_name + "/limits"][i]
        limit_uppers.append(limits.max())
        limit_lowers.append(limits.min())
    for i in range(0, all_epochs, epoch_interval):
        counts = df_hist[layer_name + "/counts"][i]
        limits = df_hist[layer_name + "/limits"][i]
        x, y = SummaryReader.histogram_to_bins(
            counts,
            limits,
            lower_bound=min(limit_lowers),
            upper_bound=max(limit_uppers),
            # n_bins=100,
        )
        df[i] = y
    df.index = x
    return df


def plot_layer_hist_for_basin_model_fold(
    model_name, chosen_layer_values, layers, bsize, cmap_str="Oranges"
):
    """_summary_

    Parameters
    ----------
    model_name : _type_
        _description_
    chosen_layer_values : _type_
        _description_
    layers : _type_
        _description_
    bsize : _type_
        _description_
    cmap_str : str, optional
        chose from here: https://matplotlib.org/stable/tutorials/colors/colormaps.html#sequential, by default "Oranges"
    """
    result_dir = os.path.join(
        definitions.RESULT_DIR,
        "tensorboard",
        "histograms",
        f"bsize{bsize}",
    )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for layer in layers:
        two_model_layers = chosen_layer_values[layer]
        try:
            df_lstm_show = two_model_layers[model_name]
        except KeyError:
            # if the model does not have this layer, skip
            continue
        lstm_x_lst = []
        lstm_y_lst = []
        lstm_dash_lines = []
        color_str = ""
        lw_lst = []
        alpha_lst = []
        cmap = cm.get_cmap(cmap_str)
        rgb_lst = []
        norm_color = colors.Normalize(vmin=0, vmax=df_lstm_show.shape[1])
        for i in df_lstm_show:
            lstm_x_lst.append(df_lstm_show.index.values)
            lstm_y_lst.append(df_lstm_show[i].values)
            lstm_dash_lines.append(True)
            color_str = color_str + "r"
            rgba = cmap(norm_color(i))
            rgb_lst.append(rgba)
            alpha_lst.append(0.5)
            lw_lst.append(0.5)
        # the first and last line should be solid, have dark color and wide width
        rgb_lst[0] = rgba
        lstm_dash_lines[-1] = False
        alpha_lst[-1] = 1
        alpha_lst[0] = 1
        lw_lst[-1] = 1
        lw_lst[0] = 1
        plot_ts(
            lstm_x_lst,
            lstm_y_lst,
            dash_lines=lstm_dash_lines,
            fig_size=(8, 4),
            # xlabel="权值",
            # ylabel="频数",
            xlabel="Weight/Bias",
            ylabel="Number",
            # c_lst=color_str,
            c_lst=rgb_lst,
            linewidth=lw_lst,
            alpha=alpha_lst,
            leg_lst=None,
        )
        plt.savefig(
            os.path.join(
                result_dir,
                f"{var}_{model_name}_{layer}_hist.png",
            ),
            dpi=600,
            bbox_inches="tight",
        )


def chosen_layer_in_layers(layers, chosen_layers):
    the_layers = []
    for layer in layers:
        the_layers.extend(
            layer for a_chosen_layer in chosen_layers if a_chosen_layer in layer
        )
    return the_layers


# plot param hist for each basin
def plot_param_hist_model(
    model_names,
    exp,
    all_epoch,
    batchsize,
    chosen_layer_for_hist,
    var_name,
    epochs_in_hist=10,
):
    chosen_layer_values = {layer: {} for layer in chosen_layer_for_hist}
    chosen_layer_values_consine = {layer: {} for layer in chosen_layer_for_hist}
    for j, model_name in tqdm(enumerate(model_names), desc=f"Variable {var_name}"):
        a_exp = exp[j]
        df_scalar, df_histgram = read_tb_log(a_exp, batchsize)
        hist_cols = df_histgram.columns.values
        model_layers = read_layer_name_from_tb_hist(hist_cols)
        chosen_layers = chosen_layer_in_layers(model_layers, chosen_layer_for_hist)
        k = 0
        for layer in chosen_layer_for_hist:
            if layer not in chosen_layers[k]:
                continue
            df_epochs_hist = epochs_hist_for_chosen_layer(
                epochs_in_hist, chosen_layers[k], df_histgram
            )
            chosen_layer_values[layer][model_name] = df_epochs_hist
            final_epoch_hist = all_epoch[j] - epochs_in_hist
            chosen_layer_values_consine[layer][model_name] = 1 - cosine(
                df_epochs_hist[0], df_epochs_hist[final_epoch_hist]
            )
            k = k + 1
    plot_layer_hist_for_basin_model_fold(
        model_names[0],
        chosen_layer_values,
        chosen_layer_for_hist,
        batchsize,
        cmap_str="Reds",
    )
    plot_layer_hist_for_basin_model_fold(
        model_names[1],
        chosen_layer_values,
        chosen_layer_for_hist,
        batchsize,
        cmap_str="Blues",
    )
    return chosen_layer_values, chosen_layer_values_consine


def merge_value(arrs_lst):
    arrs = np.array(arrs_lst)
    return np.mean(arrs, axis=0)


# NOTICE: THE ORDER CANNOT BE MODIFIED WITHOUT DEBUGGING THE CODE IN plot_param_hist_model_fold
chosen_layer_for_hist = [
    "linearIn.bias",
    "linearIn.weight",
    "lstm.b_hh",
    "lstm.b_ih",
    "lstm.w_hh",
    "lstm.w_ih",
]
# too many figures lead to "Fail to allocate bitmap"
matplotlib.use("Agg")
hist_stat_dir = os.path.join(
    definitions.RESULT_DIR,
    "hist_statistic",
)
if not os.path.exists(hist_stat_dir):
    os.makedirs(hist_stat_dir)
show_hist_bs = [100]
for i, var in enumerate(two_vars):
    for show_hist_b in show_hist_bs:
        _, chosen_layers_consine = plot_param_hist_model(
            model_names,
            exps[i],
            best_epoch[i],
            show_hist_b,
            chosen_layer_for_hist,
            var,
        )
        pd.DataFrame(chosen_layers_consine).to_csv(
            os.path.join(
                hist_stat_dir,
                f"{var}_bs{show_hist_b}_chosen_layer_consine.csv",
            )
        )

# integrate to one file for each basin
row_names = [
    "linearIn.weight",
    "linearIn.bias",
    "lstm.w_ih",
    "lstm.b_ih",
    "lstm.w_hh",
    "lstm.b_hh",
]
# NOTICE: ONLY SUPPORT two fold, two models
fold_num = 1
model_num = 2
for var in two_vars:
    basin_mat = np.full(
        (len(row_names) * fold_num, len(show_hist_bs) * model_num), np.nan
    )
    for j, show_hist_b in itertools.product(range(len(model_names)), show_hist_bs):
        cosine_sim = pd.read_csv(
            os.path.join(
                hist_stat_dir,
                f"{var}_bs{show_hist_b}_chosen_layer_consine.csv",
            ),
            index_col=0,
        )
        for i in range(len(row_names)):
            mat_row = i * fold_num
            mat_col = show_hist_bs.index(show_hist_b) * model_num + j
            basin_mat[mat_row, mat_col] = cosine_sim[row_names[i]][model_names[j]]
    pd.DataFrame(basin_mat).round(3).to_csv(
        os.path.join(hist_stat_dir, f"{var}_basin_mat.csv")
    )
