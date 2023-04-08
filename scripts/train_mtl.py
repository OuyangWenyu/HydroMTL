"""
Author: Wenyu Ouyang
Date: 2022-12-14 15:05:47
LastEditTime: 2023-04-07 20:12:32
LastEditors: Wenyu Ouyang
Description: Run multiple MTL experiments
FilePath: /HydroMTL/scripts/train_mtl.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from mtl_results_utils import run_mtl_camels_flow_et

mtl_exps = [f"exp4160{str(i + 1)}" for i in range(1)]
random_seeds = [1234, 123, 12345, 111, 1111, 11111]
for i in range(len(mtl_exps)):
    run_mtl_camels_flow_et(
        mtl_exps[i],
        random_seed=random_seeds[i],
        cache_dir=os.path.join(definitions.RESULT_DIR, "camels", mtl_exps[0]),
    )
