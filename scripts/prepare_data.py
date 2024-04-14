"""
Author: Wenyu Ouyang
Date: 2022-01-08 16:58:14
LastEditTime: 2024-04-14 18:19:15
LastEditors: Wenyu Ouyang
Description: Choose some basins for training and testing of multioutput exps
FilePath: \HydroMTL\scripts\prepare_data.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
import pandas as pd
import csv

from hydrodataset import Camels
from hydrodataset.hds_utils import usgs_screen_streamflow

from torchhydro import SETTING

# to avoid the effect from bad-quality streamflow data, we exclude sites with over 5% (about 1 year) nan values
data_dir = SETTING["local_data_path"]["datasets-origin"]
camels_dir = os.path.join(data_dir, "camels", "camels_us")
camels = Camels(camels_dir)
flow_screen_param = {"missing_data_ratio": 0.05}
selected_ids = usgs_screen_streamflow(
    camels,
    camels.read_object_ids().tolist(),
    ["2001-10-01", "2021-10-01"],
    **flow_screen_param
)
df_camels_mtl = pd.DataFrame({"GAGE_ID": selected_ids})
df_camels_mtl.to_csv(
    os.path.join("results", "camels_us_mtl_2001_2021_flow_screen.csv"),
    quoting=csv.QUOTE_NONNUMERIC,
    index=None,
)
