"""
Author: Wenyu Ouyang
Date: 2022-12-14 15:05:47
LastEditTime: 2023-04-28 15:21:30
LastEditors: Wenyu Ouyang
Description: Prepare data for k-fold cross-validation exps
FilePath: /HydroMTL/scripts/pub_prepare.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""


import csv
from functools import reduce
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from hydromtl.data.source.data_gages import Gages
from hydromtl.data.source_pro.select_gages_ids import choose_sites_in_ecoregion

selected_ids = pd.read_csv(
    os.path.join(definitions.RESULT_DIR, "camels_us_mtl_2001_2021_flow_screen.csv"),
    dtype={"GAGE_ID": str},
)["GAGE_ID"].tolist()

# kfold exps, k=2
random_seed = 1234
split_num = 2
eco_names = [
    ("ECO2_CODE", 5.2),
    ("ECO2_CODE", 5.3),
    ("ECO2_CODE", 6.2),
    ("ECO2_CODE", 7.1),
    ("ECO2_CODE", 8.1),
    ("ECO2_CODE", 8.2),
    ("ECO2_CODE", 8.3),
    ("ECO2_CODE", 8.4),
    ("ECO2_CODE", 8.5),
    ("ECO2_CODE", 9.2),
    ("ECO2_CODE", 9.3),
    ("ECO2_CODE", 9.4),
    ("ECO2_CODE", 9.5),
    ("ECO2_CODE", 9.6),
    ("ECO2_CODE", 10.1),
    ("ECO2_CODE", 10.2),
    ("ECO2_CODE", 10.4),
    ("ECO2_CODE", 11.1),
    ("ECO2_CODE", 12.1),
    ("ECO2_CODE", 13.1),
]
np.random.seed(random_seed)
kf = KFold(n_splits=split_num, shuffle=True, random_state=random_seed)
# eco attr exists in Gages
gages_data_path = os.path.join(definitions.DATASET_DIR, "gages")
gages = Gages(gages_data_path)
eco_name_chosen = []
sites_lst_train = []
sites_lst_test = []
for eco_name in eco_names:
    # chose from selected_ids
    eco_sites_id = np.array(choose_sites_in_ecoregion(gages, selected_ids, eco_name))
    if eco_sites_id.size < split_num or eco_sites_id.size < 1:
        continue
    for train, test in kf.split(eco_sites_id):
        sites_lst_train.append(eco_sites_id[train])
        sites_lst_test.append(eco_sites_id[test])
        eco_name_chosen.append(eco_name)
kfold_dir = os.path.join(
    definitions.RESULT_DIR,
    f"exp_pub_kfold{split_num}",
)
if not os.path.isdir(kfold_dir):
    os.makedirs(kfold_dir)
for i in range(split_num):
    sites_ids_train_ilst = [
        sites_lst_train[j] for j in range(len(sites_lst_train)) if j % split_num == i
    ]
    sites_ids_train_i = np.sort(
        reduce(lambda x, y: np.hstack((x, y)), sites_ids_train_ilst)
    )
    sites_ids_test_ilst = [
        sites_lst_test[j] for j in range(len(sites_lst_test)) if j % split_num == i
    ]
    sites_ids_test_i = np.sort(
        reduce(lambda x, y: np.hstack((x, y)), sites_ids_test_ilst)
    )
    kfold_i_train_df = pd.DataFrame({"GAGE_ID": sites_ids_train_i})
    kfold_i_test_df = pd.DataFrame({"GAGE_ID": sites_ids_test_i})
    kfold_i_train_df.to_csv(
        os.path.join(kfold_dir, f"camels_train_kfold{str(i)}.csv"),
        quoting=csv.QUOTE_NONNUMERIC,
        index=None,
    )
    kfold_i_test_df.to_csv(
        os.path.join(kfold_dir, f"camels_test_kfold{str(i)}.csv"),
        quoting=csv.QUOTE_NONNUMERIC,
        index=None,
    )
