"""
Author: Wenyu Ouyang
Date: 2022-01-08 16:58:14
LastEditTime: 2022-08-18 11:59:28
LastEditors: Wenyu Ouyang
Description: Choose some basins for training and testing of multioutput exps
FilePath: /HydroSPB/hydroSPB/app/multi_task/camels_multioutput_preprocess.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import KFold
from functools import reduce

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent.parent))
import definitions
from hydroSPB.data.source_pro.select_gages_ids import (
    choose_sites_in_ecoregion,
    usgs_screen_streamflow,
)
from hydroSPB.data.source.data_camels import Camels
from hydroSPB.data.source.data_gages import Gages

# to avoid the effect from bad-quality streamflow data, we exclude sites with over 5% (about 1 year) nan values
camels_dir = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
camels = Camels(camels_dir)
flow_screen_param = {"missing_data_ratio": 0.05, "zero_value_ratio": 0.999}
selected_ids = usgs_screen_streamflow(
    camels,
    camels.read_object_ids().tolist(),
    ["2001-10-01", "2021-10-01"],
    **flow_screen_param
)
df_camels_mtl = pd.DataFrame({"GAGE_ID": selected_ids})
cfg_dir = os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", "camels")
df_camels_mtl.to_csv(
    os.path.join(cfg_dir, "camels_us_mtl_2001_2021_flow_screen.csv"),
    quoting=csv.QUOTE_NONNUMERIC,
    index=None,
)

# kfold exps
random_seed = 1234
split_num = 5
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
    definitions.ROOT_DIR,
    "hydroSPB",
    "example",
    "camels",
    "exp43_kfold" + str(split_num),
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
        os.path.join(kfold_dir, "camels_train_kfold" + str(i) + ".csv"),
        quoting=csv.QUOTE_NONNUMERIC,
        index=None,
    )
    kfold_i_test_df.to_csv(
        os.path.join(kfold_dir, "camels_test_kfold" + str(i) + ".csv"),
        quoting=csv.QUOTE_NONNUMERIC,
        index=None,
    )

camels531_file = os.path.join(
    definitions.ROOT_DIR, "hydroSPB", "example", "camels531.csv"
)
camels531 = pd.read_csv(camels531_file, dtype={0: str})

final_sites = np.intersect1d(camels531["GAGE_ID"].values, selected_ids)
print(len(final_sites))

df_camels531 = pd.DataFrame({"GAGE_ID": final_sites})
cfg_dir = os.path.join(definitions.ROOT_DIR, "hydroSPB", "example", "camels")
df_camels531.to_csv(
    os.path.join(cfg_dir, "camels531_mtl_2001_2021_flow_screen.csv"),
    quoting=csv.QUOTE_NONNUMERIC,
    index=None,
)

camels_not_in_531 = [id_tmp for id_tmp in selected_ids if id_tmp not in final_sites]
print(len(camels_not_in_531))
df_camels_not_in_531 = pd.DataFrame({"GAGE_ID": camels_not_in_531})
df_camels_not_in_531.to_csv(
    os.path.join(cfg_dir, "camels_not_in_531_mtl_2001_2021_flow_screen.csv"),
    quoting=csv.QUOTE_NONNUMERIC,
    index=None,
)
