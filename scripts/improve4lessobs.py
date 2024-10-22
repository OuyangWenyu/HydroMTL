# same as notebooks/improve4var_with_less_obs.ipynb
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# Get the current directory of the project
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from scripts.mtl_results_utils import (
    plot_mtl_results_map,
)
from hydromtl.utils import hydro_constant
from hydromtl.models.trainer import stat_result
from hydromtl.utils.hydro_stat import stat_error
from scripts.streamflow_utils import get_json_file, plot_ecdf_func
from hydromtl.visual.plot_stat import plot_scatter_with_11line

exps_eval = [
    [
        "expdatascarcestlssm001",
        "expdatascarcestlssm002",
        "expdatascarcestlssm003",
        "expdatascarcestlssm004",
        "expdatascarcestlssm005",
    ],
    [
        "expdatascarcemtlqssm001",
        "expdatascarcemtlqssm002",
        "expdatascarcemtlqssm003",
        "expdatascarcemtlqssm004",
        "expdatascarcemtlqssm005",
    ],
    [
        "expdatascarcemtlqssm101",
        "expdatascarcemtlqssm102",
        "expdatascarcemtlqssm103",
        "expdatascarcemtlqssm104",
        "expdatascarcemtlqssm105",
    ],
]
var_idx = 1
q_ssm_inds = []
for exps in exps_eval:
    preds = []
    obss = []
    for i in range(len(exps)):
        cfg_dir_flow_other = os.path.join(definitions.RESULT_DIR, "camels", exps[i])
        cfg_flow_other = get_json_file(cfg_dir_flow_other)
        inds_df1, pred, obs = stat_result(
            cfg_flow_other["data_params"]["test_path"],
            cfg_flow_other["evaluate_params"]["test_epoch"],
            fill_nan=cfg_flow_other["evaluate_params"]["fill_nan"],
            var_unit=[
                hydro_constant.streamflow.unit,
                hydro_constant.surface_soil_moisture.unit,
            ],
            return_value=True,
            var_name=[
                hydro_constant.streamflow.name,
                hydro_constant.surface_soil_moisture.name,
            ],
        )
        preds.append(pred[var_idx])
        obss.append(obs[var_idx])
    pred_ensemble = np.array(preds).mean(axis=0)
    obs_ensemble = np.array(obss).mean(axis=0)
    inds_ensemble = stat_error(
        obs_ensemble,
        pred_ensemble,
        fill_nan=cfg_flow_other["evaluate_params"]["fill_nan"][var_idx],
    )
    q_ssm_inds.append(inds_ensemble["NSE"])

# ------------------------- Plots -------------------------
# Plot a Empirical Cumulative Distribution Function (ECDF) of NSE for above MTL models.
cases_exps_legends_together = [
    "STL",
    "MTL",
    "MTL_Pretrained",
]
for i in range(len(cases_exps_legends_together)):
    print(
        f"the median NSE of {cases_exps_legends_together[i]} is {np.median(q_ssm_inds[i])}"
    )
figure_dir = os.path.join(definitions.RESULT_DIR, "figures", "data_augment")
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
plot_ecdf_func(
    q_ssm_inds,
    cases_exps_legends_together=cases_exps_legends_together,
    save_path=os.path.join(
        figure_dir,
        "stl_mtl_flow_ssm_w_wo_pretrained.png",
    ),
    dash_lines=[True, True, False],
    colors=["red", "blue", "black"],
)
# plot map
gage_id_file = os.path.join(
    definitions.RESULT_DIR, "camels_us_mtl_2001_2021_flow_screen.csv"
)
basin_ids = pd.read_csv(gage_id_file, dtype={"GAGE_ID": str})
plot_mtl_results_map(
    basin_ids["GAGE_ID"].values,
    [q_ssm_inds[0], q_ssm_inds[2]],
    ["STL", "MTL_Pretrained"],
    ["o", "x"],
    os.path.join(
        figure_dir,
        "better_ssm_stl_mtl_cases_map.png",
    ),
)
# plot scatter with a 1:1 line to compare single-task and multi-task models
_, _, tesxtstr = plot_scatter_with_11line(
    q_ssm_inds[0],
    q_ssm_inds[2],
    # xlabel="NSE single-task",
    # ylabel="NSE multi-task",
    xlabel="STL NSE",
    ylabel="MTL_Pretrained NSE",
)
print(tesxtstr)
plt.savefig(
    os.path.join(
        figure_dir,
        "mtl_stl_ssm_scatter_plot_with_11line.png",
    ),
    dpi=600,
    bbox_inches="tight",
)
