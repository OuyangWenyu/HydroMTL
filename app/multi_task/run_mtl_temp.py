"""
Author: Wenyu Ouyang
Date: 2022-12-14 15:05:47
LastEditTime: 2023-01-04 21:40:16
LastEditors: Wenyu Ouyang
Description: Run MTL experiments
FilePath: /HydroSPB/hydroSPB/app/multi_task/run_mtl_temp.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
from pathlib import Path
import sys


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydroSPB.app.multi_task.mtl_results_utils import run_mtl_camels_flow_et

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
