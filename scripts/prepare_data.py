"""
Author: Wenyu Ouyang
Date: 2022-01-08 16:58:14
LastEditTime: 2023-04-07 09:24:34
LastEditors: Wenyu Ouyang
Description: Choose some basins for training and testing of multioutput exps
FilePath: /HydroMTL/scripts/prepare_data.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import pandas as pd
import csv
import sys
from pathlib import Path
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from hydromtl.data.source_pro.select_gages_ids import (
    usgs_screen_streamflow,
)
from hydromtl.data.source.data_camels import Camels

# to avoid the effect from bad-quality streamflow data, we exclude sites with over 5% (about 1 year) nan values
camels_dir = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
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
    os.path.join(definitions.RESULT_DIR, "camels_us_mtl_2001_2021_flow_screen.csv"),
    quoting=csv.QUOTE_NONNUMERIC,
    index=None,
)
