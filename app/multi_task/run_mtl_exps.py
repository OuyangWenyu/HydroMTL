"""
Author: Wenyu Ouyang
Date: 2022-12-14 15:05:47
LastEditTime: 2023-02-09 14:54:13
LastEditors: Wenyu Ouyang
Description: Run MTL experiments
FilePath: /HydroSPB/hydroSPB/app/multi_task/run_mtl_exps.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
from pathlib import Path
import sys


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydroSPB.app.multi_task.mtl_results_utils import run_mtl_camels_flow_et

mtl_exps = [f"exp4160{str(i + 1)}" for i in range(1)]
random_seeds = [1234, 123, 12345, 111, 1111, 11111]
# for i in range(len(mtl_exps)):
#     run_mtl_camels_flow_et(
#         mtl_exps[i],
#         random_seed=random_seeds[i],
#         cache_dir=os.path.join(
#             definitions.ROOT_DIR, "hydroSPB", "example", "camels", mtl_exps[0]
#         ),
#     )

mtl_uw_exps = [f"exp4170{str(i + 1)}" for i in range(1)]
run_mtl_camels_flow_et(
    mtl_uw_exps[0],
    random_seed=random_seeds[0],
    cache_dir=os.path.join(
        definitions.ROOT_DIR, "hydroSPB", "example", "camels", mtl_uw_exps[0]
    ),
    loss_func="UncertaintyWeights",
)

# TG7 basins: 60668,61561,63002,63007,63486,92354,94560
# basin_ids = ["60668", "61561", "63002", "63007", "63486", "92354", "94560"]
# mtl_exps = [f"exp6110{str(i + 1)}" for i in range(6)]
# random_seeds = [1234, 123, 12345, 111, 1111, 11111]
# for i in range(len(mtl_exps)):
#     test_mtl_camels_cc_flow_et(
#         mtl_exps[i],
#         random_seed=random_seeds[i],
#         freeze_params=None,
#         opt="Adadelta",
#         # opt_param={"lr": 0.5},
#         batch_size=5,
#         epoch=10,
#         save_epoch=1,
#         gage_id=basin_ids,
#         # data_loader="StreamflowDataset",
#     )
