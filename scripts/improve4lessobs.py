# same as notebooks/improve4var_with_less_obs.ipynb
import os
import sys
from matplotlib import pyplot as plt
import pandas as pd

# Get the current directory of the project
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
import definitions
from scripts.mtl_results_utils import (
    read_multi_single_exps_results,
    plot_mtl_results_map,
)
from hydromtl.utils import hydro_constant
from scripts.streamflow_utils import plot_ecdf_func
from hydromtl.visual.plot_stat import plot_scatter_with_11line

# ------------------- training -------------------
# Some commands to train the model:
# # STL Q
# python run_task.py --exp expstlq201 --output_vars usgsFlow ssm --loss_weight 1.0 0.0 --train_period 2005-10-01 2015-10-01 --test_period 2015-10-01 2018-10-01 --ctx 0 --random 1234 --limit_part 1
# # STL SSM
# python run_task.py --exp expstlssm001 --output_vars usgsFlow ssm --loss_weight 0.0 1.0 --train_period 2015-10-01 2018-10-01 --test_period 2018-10-01 2021-10-01 --ctx 1 --random 1234 --limit_part 0
# # MTL without STL-Q-pretrained
# python run_task.py --exp expmtlqssm001 --output_vars usgsFlow ssm --loss_weight 0.5 0.5 --train_period 2015-10-01 2018-10-01 --test_period 2018-10-01 2021-10-01 --ctx 0 --random 1234
# # MTL with STL-Q-pretrained; test_epoch is same as train_epoch
# python run_task.py --exp expmtlqssm101 --output_vars usgsFlow ssm --loss_weight 0.5 0.5 --train_period 2015-10-01 2018-10-01 --test_period 2018-10-01 2021-10-01 --ctx 1 --random 1234 --weight_path /mnt/sdc/owen/code/HydroMTL/results/camels/expstlq201/model_Ep200.pth --train_epoch 100

# ------------------- evaluation ------------------
# After run all above commands, the results are saved in the folder `results/camels/`. The results are shown in the following figure.
exps_eval = ["expstlssm001", "expmtlqssm001", "expmtlqssm101"]
q_ssm_inds, _ = read_multi_single_exps_results(
    exps_eval,
    var_idx=1,
    ensemble=-1,
    var_names=[
        hydro_constant.streamflow.name,
        hydro_constant.surface_soil_moisture.name,
    ],
    var_units=["ft3/s", "mm/day"],
)

# ------------------------- Plots -------------------------
# Plot a Empirical Cumulative Distribution Function (ECDF) of NSE for above MTL models.
cases_exps_legends_together = [
    "STL",
    "MTL",
    "MTL_Pretrained",
]
figure_dir = os.path.join(definitions.RESULT_DIR, "figures")
plot_ecdf_func(
    q_ssm_inds[:-1],
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
plot_scatter_with_11line(
    q_ssm_inds[0],
    q_ssm_inds[2],
    # xlabel="NSE single-task",
    # ylabel="NSE multi-task",
    xlabel="STL NSE",
    ylabel="MTL_Pretrained NSE",
)
plt.savefig(
    os.path.join(
        figure_dir,
        "mtl_stl_ssm_scatter_plot_with_11line.png",
    ),
    dpi=600,
    bbox_inches="tight",
)
