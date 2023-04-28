"""
Author: Wenyu Ouyang
Date: 2022-12-14 15:05:47
LastEditTime: 2023-04-27 22:14:39
LastEditors: Wenyu Ouyang
Description: Run multiple STL experiments
FilePath: /HydroMTL/scripts/train_stl.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path
import sys


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from scripts.mtl_results_utils import run_mtl_camels_flow_et

stl_exps = [f"expstlq00{str(i + 1)}" for i in range(1)]
# stl_exps = [f"expstlet00{str(i + 1)}" for i in range(1)]
random_seeds = [1234, 123, 12345, 111, 1111, 11111]
for i in range(len(stl_exps)):
    run_mtl_camels_flow_et(
        stl_exps[i],
        random_seed=random_seeds[i],
        cache_dir=os.path.join(definitions.RESULT_DIR, "camels", stl_exps[0]),
        ctx=[1],
        # streamflow
        weight_ratio=[1, 0],
        limit_part=[1],
        # et
        # weight_ratio=[0, 1],
        # limit_part=[0],
    )
