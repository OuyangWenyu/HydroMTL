"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2023-05-10 08:58:57
LastEditors: Wenyu Ouyang
Description: Just try some code
FilePath: /HydroMTL/scripts/try.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from mtl_results_utils import stat_mtl_1var_ensemble_result
from hydromtl.utils import hydro_constant

stl_test_exps = ["exppub1010", "exppub1020"]
var_names = [hydro_constant.streamflow.name, hydro_constant.evapotranspiration.name]
var_units = [hydro_constant.streamflow.unit, hydro_constant.evapotranspiration.unit]
var_idx = 1
(
    inds_df_stl_test,
    pred_mean_stl_test,
    obs_mean_stl_test,
    all_inds_stl_test,
) = stat_mtl_1var_ensemble_result(
    stl_test_exps,
    var_names=var_names,
    var_units=var_units,
    return_value=True,
    var_idx=var_idx,
)
