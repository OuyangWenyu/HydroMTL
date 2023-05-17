"""
Author: Wenyu Ouyang
Date: 2023-05-16 20:48:04
LastEditTime: 2023-05-17 11:27:25
LastEditors: Wenyu Ouyang
Description: Just try some code
FilePath: /HydroMTL/scripts/try.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydromtl.data.source import data_constant
from hydromtl.explain.probe_analysis import show_probe

import time

start = time.time()
save_dir = os.path.join(
    definitions.RESULT_DIR,
    "figures",
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
run_exp_lst = [
    f"camels{os.sep}expstlq0010",
    f"camels{os.sep}expmtl0030",
    f"camels{os.sep}expstlet0010",
]
legend_lst = ["STL-Q", "MTL", "STL-ET"]

show_probe(
    run_exp_lst=run_exp_lst,
    var=data_constant.surface_soil_moisture_smap_camels_us,
    legend_lst=legend_lst,
    show_probe_metric="Corr",
    retrian_probe=[False, False, False],
    num_workers=0,
    save_dir=save_dir,
)
end = time.time()

print(f"The code took {end - start} seconds to run.")
# show_probe(
#     run_exp_lst=run_exp_lst,
#     var=data_constant.evapotranspiration_modis_camels_us,
#     legend_lst=legend_lst,
#     show_probe_metric="Corr",
#     retrian_probe=[False, False, False],
#     num_workers=0,
#     save_dir=save_dir,
# )

# show_probe(
#     run_exp_lst=run_exp_lst,
#     var=data_constant.streamflow_camels_us,
#     legend_lst=legend_lst,
#     show_probe_metric="Corr",
#     retrian_probe=[False, False, False],
#     num_workers=0,
#     save_dir=save_dir,
# )
