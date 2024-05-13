# Scaling experiments
# Here we try to plot figures of scaling curves for MTL models and STL models and their PUB results. It can reflect how models perform for temporal and spatial generalization when the number of trained basins increases.
# When the number of basins is small, the choice of basins may be biased. So we randomly select basins from the whole dataset for many times and calculate the mean/median metrics as the final result.
# We can see the difference between MTL and STL models.

import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import pickle

# Get the current directory of the repo
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from scripts.streamflow_utils import get_json_file
from hydromtl.models.trainer import stat_result
from hydromtl.utils import hydro_constant
from hydromtl.visual.plot_stat import plot_simple_lines

x_percents = [5, 10, 20, 25, 33, 50, 66, 75, 80, 90, 95]

# There are multiple sub-experiments in each x_percent setting. Read them all and calcuate the mean/median value of all sub-experiments as the final result.
# First, get the list of all experiments.
mtl_scale_exps = [
    f"expscalemtlpercent{str(x_percents[i]).zfill(3)}" for i in range(len(x_percents))
]
et_scale_exps = [
    f"expscalestletpercent{str(x_percents[i]).zfill(3)}" for i in range(len(x_percents))
]
q_scale_exps = [
    f"expscalestlqpercent{str(x_percents[i]).zfill(3)}" for i in range(len(x_percents))
]

# Then, for each experiment, get the list of all sub-experiments.
# 1. Temporal generalization sub-experiments
all_mtl_scale_temporal_exps = []
all_et_scale_temporal_exps = []
all_q_scale_temporal_exps = []
for i in range(len(mtl_scale_exps)):
    x_percent = x_percents[i]
    if x_percent < 50:
        split_num = int(1 / (x_percent / 100))
    else:
        split_num = round(1 / (1 - x_percent / 100))
    mtl_scale_temporal_exps = [
        mtl_scale_exps[i] + str(j + 1).zfill(2) for j in range(split_num)
    ]
    et_scale_temporal_exps = [
        et_scale_exps[i] + str(j + 1).zfill(2) for j in range(split_num)
    ]
    q_scale_temporal_exps = [
        q_scale_exps[i] + str(j + 1).zfill(2) for j in range(split_num)
    ]
    all_mtl_scale_temporal_exps.append(mtl_scale_temporal_exps)
    all_et_scale_temporal_exps.append(et_scale_temporal_exps)
    all_q_scale_temporal_exps.append(q_scale_temporal_exps)


def read_metric_of_all_sub_exps(
    scale_exps,
    var_idx=0,
    metric="NSE",
    var_names=None,
    var_units=None,
    save_file="list.pkl",
):
    if var_names is None:
        var_names = [
            hydro_constant.streamflow.name,
            hydro_constant.evapotranspiration.name,
        ]
    if var_units is None:
        var_units = ["ft3/s", "mm/day"]
    # Check if the saved list file exists
    if os.path.exists(save_file):
        # Load the list from the file
        with open(save_file, "rb") as file:
            ind_all_lst = pickle.load(file)
    else:
        ind_all_lst = []
        for i in range(len(scale_exps)):
            # for each sub-experiment
            ind_lst = []
            for j in range(len(scale_exps[i])):
                cfg_dir = os.path.join(
                    definitions.RESULT_DIR,
                    "camels",
                    scale_exps[i][j],
                )
                try:
                    cfg_data = get_json_file(cfg_dir)
                    inds_df_tmp, _, _ = stat_result(
                        cfg_data["data_params"]["test_path"],
                        cfg_data["evaluate_params"]["test_epoch"],
                        fill_nan=cfg_data["evaluate_params"]["fill_nan"],
                        return_value=True,
                        var_unit=var_units,
                        var_name=var_names,
                    )
                except Exception:
                    print(f"Error in reading {cfg_dir}")
                    continue
                ind_lst.append(inds_df_tmp[var_idx][metric].values)
            ind_all_lst.append(ind_lst)
        with open(save_file, "wb") as file:
            pickle.dump(ind_all_lst, file)
    return ind_all_lst


# Then, we can read results of all the sub-experiments
cache_dir = os.path.join(definitions.RESULT_DIR, "cache")
# For streamflow of MTL exps:
mtlq_temporal_metric_lst = read_metric_of_all_sub_exps(
    all_mtl_scale_temporal_exps,
    save_file=os.path.join(cache_dir, "mtlq_temporal_metric_lst.pkl"),
)
# For streamflow of STL exps:
stlq_temporal_metric_lst = read_metric_of_all_sub_exps(
    all_q_scale_temporal_exps,
    save_file=os.path.join(cache_dir, "stlq_temporal_metric_lst.pkl"),
)
# For evapotranspiration of MTL exps:
mtlet_temporal_metric_lst = read_metric_of_all_sub_exps(
    all_mtl_scale_temporal_exps,
    var_idx=1,
    save_file=os.path.join(cache_dir, "mtlet_temporal_metric_lst.pkl"),
)
# For evapotranspiration of STL exps:
stlet_temporal_metric_lst = read_metric_of_all_sub_exps(
    all_et_scale_temporal_exps,
    var_idx=1,
    save_file=os.path.join(cache_dir, "stlet_temporal_metric_lst.pkl"),
)

# 2. Spatial generalization sub-experiments
all_mtl_scale_spatial_exps = []
all_q_scale_spatial_exps = []
all_et_scale_spatial_exps = []
for i in range(len(mtl_scale_exps)):
    x_percent = x_percents[i]
    if x_percent < 50:
        split_num = int(1 / (x_percent / 100))
    else:
        split_num = round(1 / (1 - x_percent / 100))
    mtl_scale_spatial_exps = [
        mtl_scale_exps[i] + str(j + 1).zfill(2) + "0" for j in range(split_num)
    ]
    q_scale_spatial_exps = [
        q_scale_exps[i] + str(j + 1).zfill(2) + "0" for j in range(split_num)
    ]
    et_scale_spatial_exps = [
        et_scale_exps[i] + str(j + 1).zfill(2) + "0" for j in range(split_num)
    ]
    all_mtl_scale_spatial_exps.append(mtl_scale_spatial_exps)
    all_q_scale_spatial_exps.append(q_scale_spatial_exps)
    all_et_scale_spatial_exps.append(et_scale_spatial_exps)

# Read metrics:
mtlq_spatial_metric_lst = read_metric_of_all_sub_exps(
    all_mtl_scale_spatial_exps,
    save_file=os.path.join(cache_dir, "mtlq_spatial_metric_lst.pkl"),
)
stlq_spatial_metric_lst = read_metric_of_all_sub_exps(
    all_q_scale_spatial_exps,
    save_file=os.path.join(cache_dir, "stlq_spatial_metric_lst.pkl"),
)
mtlet_spatial_metric_lst = read_metric_of_all_sub_exps(
    all_mtl_scale_spatial_exps,
    var_idx=1,
    save_file=os.path.join(cache_dir, "mtlet_spatial_metric_lst.pkl"),
)
stlet_spatial_metric_lst = read_metric_of_all_sub_exps(
    all_et_scale_spatial_exps,
    var_idx=1,
    save_file=os.path.join(cache_dir, "stlet_spatial_metric_lst.pkl"),
)


# ----------------------------- Plots ------------------------------
# Next, we can plot the scaling curves.
# Temporal exps are plotted in one figure and spatial exps are plotted in another figure.
def mean_of_medians(metric_lst):
    medians = []
    for arr_lst in metric_lst:
        a_median = [np.median(arr) for arr in arr_lst]
        medians.append(np.mean(a_median))
    return medians


xlabel = "Percent basin used for training"
ylabel = "Median NSE"
figure_dir = os.path.join(definitions.RESULT_DIR, "figures")
mtlq_temporal_medians = mean_of_medians(mtlq_temporal_metric_lst)
stlq_temporal_medians = mean_of_medians(stlq_temporal_metric_lst)

mtlet_temporal_medians = mean_of_medians(mtlet_temporal_metric_lst)
stlet_temporal_medians = mean_of_medians(stlet_temporal_metric_lst)

mtlq_spatial_medians = mean_of_medians(mtlq_spatial_metric_lst)
stlq_spatial_medians = mean_of_medians(stlq_spatial_metric_lst)


mtlet_spatial_medians = mean_of_medians(mtlet_spatial_metric_lst)
stlet_spatial_medians = mean_of_medians(stlet_spatial_metric_lst)

# Try to plot the scaling curves of temporal and spatial generalization of streamflow (evapotranspiration) in one figure.
plot_simple_lines(
    [x_percents, x_percents, x_percents, x_percents],
    [
        mtlq_temporal_medians,
        stlq_temporal_medians,
        mtlq_spatial_medians,
        stlq_spatial_medians,
    ],
    legends=["MTL Q", "STL Q", "MTL PUB Q", "STL PUB Q"],
    x_str=xlabel,
    y_str=ylabel,
    dash_lines=[False, True, False, True],
    colors="rrbb",
)
plt.savefig(os.path.join(figure_dir, "scale_q.png"), dpi=600, bbox_inches="tight")
plot_simple_lines(
    [x_percents, x_percents, x_percents, x_percents],
    [
        mtlet_temporal_medians,
        stlet_temporal_medians,
        mtlet_spatial_medians,
        stlet_spatial_medians,
    ],
    legends=["MTL ET", "STL ET", "MTL PUB ET", "STL PUB ET"],
    x_str=xlabel,
    y_str=ylabel,
    dash_lines=[False, True, False, True],
    colors="rrbb",
)
plt.savefig(os.path.join(figure_dir, "scale_et.png"), dpi=600, bbox_inches="tight")
