"""
Author: Wenyu Ouyang
Date: 2022-12-14 15:05:47
LastEditTime: 2023-04-06 21:49:39
LastEditors: Wenyu Ouyang
Description: Run MTL experiments
FilePath: /HydroMTL/hydromtl/scripts/run_mtl_temp.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
from pathlib import Path
import sys


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydromtl.scripts.mtl_results_utils import run_mtl_camels_flow_et

mtl_exps = [f"exp4160{str(i + 1)}" for i in range(2)]
random_seeds = [1234, 123, 12345, 111, 1111, 11111]
for i in range(1, len(mtl_exps)):
    run_mtl_camels_flow_et(
        mtl_exps[i],
        random_seed=random_seeds[i],
        cache_dir=os.path.join(
            definitions.ROOT_DIR, "hydroSPB", "example", "camels", mtl_exps[0]
        ),
        ctx=[1]
    )
